#
# Localization Module for Local Refinement (v2)
# Implements multi-strategy 2D-3D precise mapping:
#   Strategy 1: Ray triangulation (from GS-LPM)
#   Strategy 2: Depth backprojection (from CL-Splats)
#   Strategy 3: Projection contribution statistics (from FlashSplat idea)
#   Strategy 4: Masked-loss gradient attribution (v1, fixed)
# Plus: multi-strategy fusion, multi-view voting, 3D clustering.
#
# v2 changes:
#   - Vectorized projection (no per-Gaussian loops)
#   - Ray triangulation via camera params
#   - Depth backprojection + kNN
#   - Contribution statistics via vectorized overlap
#   - Fix GroupParams access via getattr
#   - GPU-accelerated distance computation for clustering
#

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
import cv2


def _getparam(params, name, default):
    """Safely get a parameter from either a GroupParams object or a dict."""
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


# ===========================================================================
# Projection utilities (vectorized)
# ===========================================================================

def project_gaussians_to_2d(
    gaussians,
    viewpoint_cam,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project Gaussian centers to 2D pixel coordinates (fully vectorized).

    Returns:
        xy_pixel: [N, 2] pixel coordinates (x, y)
        depth: [N] depth values in camera space
    """
    xyz = gaussians.get_xyz  # [N, 3]
    N = xyz.shape[0]

    ones = torch.ones(N, 1, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)  # [N, 4]

    # World to camera
    view_mat = viewpoint_cam.world_view_transform  # [4, 4]
    xyz_cam = xyz_h @ view_mat  # [N, 4]
    depth = xyz_cam[:, 2]

    # Full projection
    proj_mat = viewpoint_cam.full_proj_transform  # [4, 4]
    xyz_proj = xyz_h @ proj_mat  # [N, 4]

    w = xyz_proj[:, 3].clamp(min=1e-6)
    ndc_x = xyz_proj[:, 0] / w
    ndc_y = xyz_proj[:, 1] / w

    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    px = ((ndc_x + 1.0) * 0.5) * W
    py = ((ndc_y + 1.0) * 0.5) * H

    xy_pixel = torch.stack([px, py], dim=1)
    return xy_pixel, depth


def get_camera_position(viewpoint_cam) -> torch.Tensor:
    """Extract camera world position from view transform. Returns [3] tensor."""
    # world_view_transform is column-major: cam2world = inverse(view_mat)
    # For a standard 4x4 view matrix V, camera position = -R^T * t
    view_mat = viewpoint_cam.world_view_transform.T  # now row-major [4, 4]
    R = view_mat[:3, :3]
    t = view_mat[:3, 3]
    cam_pos = -R.T @ t
    return cam_pos


def get_camera_direction(viewpoint_cam) -> torch.Tensor:
    """Get camera forward direction (world space). Returns [3] unit vector."""
    view_mat = viewpoint_cam.world_view_transform.T
    R = view_mat[:3, :3]
    # Camera looks along -Z in camera space → forward = -R^T * [0,0,1]
    forward = -R.T @ torch.tensor([0., 0., 1.], device=view_mat.device)
    return F.normalize(forward, dim=0)


# ===========================================================================
# Strategy 1: Ray triangulation (from GS-LPM)
# ===========================================================================

def compute_camera_rays(
    viewpoint_cam,
    pixel_coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute world-space ray origin and direction for given pixel coordinates.
    (From GS-LPM set_rays_od)

    Args:
        viewpoint_cam: Camera object
        pixel_coords: [M, 2] pixel coordinates (x, y)

    Returns:
        origins: [M, 3] ray origins (camera position, repeated)
        directions: [M, 3] ray directions (world space, normalized)
    """
    device = pixel_coords.device
    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height

    # Camera position
    cam_pos = get_camera_position(viewpoint_cam)  # [3]

    # Pixel to NDC
    ndc_x = (pixel_coords[:, 0] / W) * 2.0 - 1.0
    ndc_y = (pixel_coords[:, 1] / H) * 2.0 - 1.0

    M = pixel_coords.shape[0]

    # NDC to camera space via inverse projection
    proj_mat = viewpoint_cam.full_proj_transform.T  # row-major
    try:
        proj_inv = torch.linalg.inv(proj_mat)
    except Exception:
        proj_inv = torch.inverse(proj_mat)

    # Homogeneous NDC points at z=1
    ndc_pts = torch.stack([ndc_x, ndc_y, torch.ones(M, device=device),
                           torch.ones(M, device=device)], dim=1)  # [M, 4]
    cam_pts = ndc_pts @ proj_inv.T  # [M, 4]
    cam_pts = cam_pts[:, :3] / cam_pts[:, 3:4].clamp(min=1e-6)

    # Camera to world
    view_mat = viewpoint_cam.world_view_transform.T  # row-major
    try:
        view_inv = torch.linalg.inv(view_mat)
    except Exception:
        view_inv = torch.inverse(view_mat)

    R_inv = view_inv[:3, :3]
    directions = (cam_pts @ R_inv.T)  # [M, 3]
    directions = F.normalize(directions, dim=1)

    origins = cam_pos.unsqueeze(0).expand(M, 3)
    return origins, directions


def find_paired_views(
    views: list,
    ref_idx: int,
    angle_min: float = 15.0,
    angle_max: float = 60.0,
    max_pairs: int = 5,
) -> List[int]:
    """Find views paired with ref_idx based on viewing angle.
    (From GS-LPM get_paired_views)

    Args:
        views: List of Camera objects
        ref_idx: Reference view index
        angle_min/max: Acceptable angle range (degrees) between view directions
        max_pairs: Maximum number of pairs to return

    Returns:
        List of paired view indices, sorted by angle proximity to midpoint
    """
    ref_dir = get_camera_direction(views[ref_idx])
    ref_pos = get_camera_position(views[ref_idx])

    candidates = []
    for i, v in enumerate(views):
        if i == ref_idx:
            continue
        v_dir = get_camera_direction(v)
        cos_angle = (ref_dir * v_dir).sum().clamp(-1, 1)
        angle_deg = torch.acos(cos_angle).item() * 180.0 / math.pi
        if angle_min <= angle_deg <= angle_max:
            # Prefer angles near the middle of the range
            mid = (angle_min + angle_max) / 2.0
            candidates.append((i, abs(angle_deg - mid)))

    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates[:max_pairs]]


def triangulate_3d_zones(
    view1,
    view2,
    regions1: List[Dict],
    regions2: List[Dict],
    defect_mask1: torch.Tensor,
    defect_mask2: torch.Tensor,
) -> List[Dict]:
    """Triangulate 3D zones from defect regions in two views.
    (From GS-LPM zones3d_projection)

    For each region in view1, shoot rays from the region centroid and corners.
    Find corresponding regions in view2 and triangulate to get 3D bounding boxes.

    Returns:
        List of zone dicts with "center" [3] and "radius" float
    """
    if not regions1 or not regions2:
        return []

    device = defect_mask1.device
    zones = []

    pos1 = get_camera_position(view1)
    pos2 = get_camera_position(view2)

    for r1 in regions1:
        x1, y1, w1, h1 = r1["bbox"]
        cx1, cy1 = r1["centroid"]

        # Shoot ray from view1 through region centroid
        pt1 = torch.tensor([[cx1, cy1]], device=device, dtype=torch.float32)
        _, dir1 = compute_camera_rays(view1, pt1)
        dir1 = dir1[0]  # [3]

        # Find closest region in view2 by projecting view1 centroid direction
        best_r2 = None
        best_dist = float('inf')
        for r2 in regions2:
            cx2, cy2 = r2["centroid"]
            pt2 = torch.tensor([[cx2, cy2]], device=device, dtype=torch.float32)
            _, dir2 = compute_camera_rays(view2, pt2)
            dir2 = dir2[0]

            # Closest point between two rays (midpoint method)
            w0 = pos1 - pos2
            a = (dir1 * dir1).sum()
            b = (dir1 * dir2).sum()
            c = (dir2 * dir2).sum()
            d = (dir1 * w0).sum()
            e = (dir2 * w0).sum()
            denom = a * c - b * b
            if abs(denom) < 1e-8:
                continue
            sc = (b * e - c * d) / denom
            tc = (a * e - b * d) / denom

            p1 = pos1 + sc * dir1
            p2 = pos2 + tc * dir2
            center = (p1 + p2) / 2.0
            dist = (p1 - p2).norm().item()

            if dist < best_dist:
                best_dist = dist
                best_r2 = r2
                best_center = center

        if best_r2 is not None and best_dist < 100.0:  # sanity check
            # Radius: proportional to region size and triangulation error
            diag1 = math.sqrt(w1**2 + h1**2)
            scale = diag1 / max(view1.image_width, view1.image_height)
            radius = max(best_dist * 2.0, scale * 10.0)
            zones.append({
                "center": best_center.detach(),
                "radius": radius,
            })

    return zones


def find_gaussians_in_zones(
    gaussians,
    zones: List[Dict],
) -> torch.Tensor:
    """Find Gaussians inside 3D zones (spheres).
    (From GS-LPM get_points_in_cones — simplified to spheres)

    Returns:
        scores: [N] float, > 0 means inside at least one zone
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    xyz = gaussians.get_xyz.detach()  # [N, 3]
    scores = torch.zeros(N, device=device)

    for zone in zones:
        center = zone["center"].to(device)  # [3]
        radius = zone["radius"]
        dists = (xyz - center.unsqueeze(0)).norm(dim=1)  # [N]
        inside = dists <= radius
        # Score: inverse distance, higher for closer Gaussians
        zone_score = torch.where(inside, 1.0 - dists / (radius + 1e-8), torch.zeros_like(dists))
        scores = torch.max(scores, zone_score)

    return scores


# ===========================================================================
# Strategy 2: Depth backprojection (from CL-Splats)
# ===========================================================================

def depth_backproject_to_gaussians(
    viewpoint_cam,
    defect_mask: torch.Tensor,
    depth_image: torch.Tensor,
    gaussians,
    knn_k: int = 3,
    local_radius: float = 3.0,
    max_pixels: int = 5000,
) -> torch.Tensor:
    """Backproject defect pixels using depth, then find nearest Gaussians via kNN.
    (From CL-Splats depth_anything_lifter.py::lift)

    Args:
        viewpoint_cam: Camera object
        defect_mask: [H, W] bool
        depth_image: [1, H, W] or [H, W] rendered depth
        gaussians: GaussianModel
        knn_k: Number of nearest neighbors
        local_radius: Scale-aware radius multiplier
        max_pixels: Max defect pixels to process (subsample if more)

    Returns:
        scores: [N] float, positive evidence per Gaussian
    """
    device = defect_mask.device
    N = gaussians.get_xyz.shape[0]
    scores = torch.zeros(N, device=device)

    if depth_image is None:
        return scores

    if depth_image.dim() == 3:
        depth_image = depth_image.squeeze(0)  # [H, W]

    H, W = defect_mask.shape

    # Get defect pixel coordinates
    ys, xs = torch.where(defect_mask)
    if len(ys) == 0:
        return scores

    # Subsample if too many pixels
    if len(ys) > max_pixels:
        perm = torch.randperm(len(ys), device=device)[:max_pixels]
        ys, xs = ys[perm], xs[perm]

    depths = depth_image[ys, xs]
    valid = depths > 0
    ys, xs, depths = ys[valid], xs[valid], depths[valid]
    if len(ys) == 0:
        return scores

    # Backproject to 3D: pixel + depth -> camera coords -> world coords
    view_mat = viewpoint_cam.world_view_transform.T  # row-major [4, 4]
    proj_mat = viewpoint_cam.full_proj_transform.T

    try:
        proj_inv = torch.linalg.inv(proj_mat)
    except Exception:
        proj_inv = torch.inverse(proj_mat)
    try:
        view_inv = torch.linalg.inv(view_mat)
    except Exception:
        view_inv = torch.inverse(view_mat)

    ndc_x = (xs.float() / W) * 2.0 - 1.0
    ndc_y = (ys.float() / H) * 2.0 - 1.0
    M = len(ndc_x)

    ndc_pts = torch.stack([ndc_x, ndc_y,
                           torch.ones(M, device=device),
                           torch.ones(M, device=device)], dim=1)
    cam_pts = ndc_pts @ proj_inv.T
    cam_pts = cam_pts[:, :3] / cam_pts[:, 3:4].clamp(min=1e-6)

    # Scale by actual depth
    cam_pts_norm = cam_pts.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cam_pts = cam_pts / cam_pts_norm * depths.unsqueeze(1)

    # Camera to world
    ones = torch.ones(M, 1, device=device)
    cam_pts_h = torch.cat([cam_pts, ones], dim=1)  # [M, 4]
    world_pts = cam_pts_h @ view_inv.T  # [M, 4]
    world_pts = world_pts[:, :3]  # [M, 3]

    # kNN: find nearest Gaussians for each backprojected point
    xyz = gaussians.get_xyz.detach()  # [N, 3]
    scales = gaussians.get_scaling.detach()  # [N, 3]
    avg_scale = scales.mean(dim=1)  # [N]

    # Process in chunks to avoid OOM: each row of cdist is N*4 bytes.
    # With N=3.9M, one row ≈ 15MB, so cap chunks to fit in available VRAM.
    bytes_per_row = N * 4
    free_mem = torch.cuda.mem_get_info()[0] if hasattr(torch.cuda, 'mem_get_info') else 4e9
    safe_chunk = max(16, int(free_mem * 0.3 / bytes_per_row))
    chunk_size = min(safe_chunk, M)

    scale_denom = avg_scale * local_radius + 1e-8  # [N], precompute once
    k = min(knn_k, N)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        pts_chunk = world_pts[start:end]  # [C, 3]

        dists = torch.cdist(pts_chunk, xyz)  # [C, N]
        dists.div_(scale_denom.unsqueeze(0))  # in-place scale-aware

        _, knn_idx = dists.topk(k, dim=1, largest=False)  # [C, k]
        del dists

        # Accumulate positive evidence
        for j in range(k):
            weight = 1.0 / (j + 1)
            indices = knn_idx[:, j]
            scores.scatter_add_(0, indices, torch.full_like(indices, weight, dtype=torch.float32))

    return scores


# ===========================================================================
# Strategy 3: Contribution statistics (from FlashSplat idea, vectorized)
# ===========================================================================

def compute_contribution_scores(
    gaussians,
    viewpoint_cam,
    defect_mask: torch.Tensor,
    render_pkg: dict,
    min_overlap: float = 0.1,
) -> torch.Tensor:
    """Approximate per-Gaussian contribution to defect pixels via projection overlap.
    (Inspired by FlashSplat used_count, without modifying CUDA rasterizer)

    Vectorized: no per-Gaussian loops.

    Returns:
        scores: [N] float contribution scores
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    H, W = defect_mask.shape

    xy_pixel, depth = project_gaussians_to_2d(gaussians, viewpoint_cam)
    radii = render_pkg.get("radii", None)
    if radii is None:
        return torch.zeros(N, device=device)

    visible = radii > 0
    scores = torch.zeros(N, device=device)

    # Pre-compute cumulative defect mask for fast area lookup
    # Use integral image for O(1) rectangle sum queries
    mask_np = defect_mask.float().cpu().numpy()
    integral = cv2.integral(mask_np)  # [H+1, W+1]
    integral_t = torch.from_numpy(integral).float().to(device)

    # Process only visible Gaussians
    vis_idx = visible.nonzero(as_tuple=True)[0]
    if len(vis_idx) == 0:
        return scores

    px = xy_pixel[vis_idx, 0]  # [V]
    py = xy_pixel[vis_idx, 1]
    r = radii[vis_idx].float()  # [V]

    # Bounding boxes
    x_min = (px - r).long().clamp(0, W - 1)
    x_max = (px + r + 1).long().clamp(1, W)
    y_min = (py - r).long().clamp(0, H - 1)
    y_max = (py + r + 1).long().clamp(1, H)

    # Vectorized integral image lookup
    # sum = I(y2, x2) - I(y1, x2) - I(y2, x1) + I(y1, x1)
    defect_area = (integral_t[y_max, x_max] - integral_t[y_min, x_max]
                   - integral_t[y_max, x_min] + integral_t[y_min, x_min])
    total_area = ((x_max - x_min) * (y_max - y_min)).float().clamp(min=1.0)
    overlap_ratio = defect_area / total_area

    # Score = overlap ratio (higher = more contribution to defect)
    vis_scores = torch.where(overlap_ratio >= min_overlap, overlap_ratio, torch.zeros_like(overlap_ratio))
    scores[vis_idx] = vis_scores

    return scores


# ===========================================================================
# Strategy 4: Gradient attribution (v1, fixed)
# ===========================================================================

def compute_gradient_attribution(
    gaussians,
    viewpoint_cam,
    defect_mask: torch.Tensor,
    render_fn,
    pipe,
    background: torch.Tensor,
    attr_weights,
    separate_sh: bool = False,
    use_trained_exp: bool = False,
) -> torch.Tensor:
    """Compute per-Gaussian attribution score via masked-loss backpropagation.

    Args:
        attr_weights: GroupParams or dict with attr_w_xyz, attr_w_feat, etc.

    Returns:
        scores: [N] per-Gaussian attribution score
    """
    # Zero existing gradients. Optimizer may not exist during analysis-only mode.
    if hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
        gaussians.optimizer.zero_grad(set_to_none=True)
    else:
        for p in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                   gaussians._opacity, gaussians._scaling, gaussians._rotation]:
            if p.grad is not None:
                p.grad.zero_()

    render_pkg = render_fn(
        viewpoint_cam, gaussians, pipe, background,
        use_trained_exp=use_trained_exp,
        separate_sh=separate_sh
    )
    rendered = render_pkg["render"]
    gt = viewpoint_cam.original_image[:3, :, :].cuda()

    if viewpoint_cam.alpha_mask is not None:
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        rendered = rendered * alpha_mask

    mask_float = defect_mask.float().unsqueeze(0)
    masked_diff = (rendered - gt) * mask_float
    local_loss = masked_diff.abs().sum() / (mask_float.sum() * 3 + 1e-8)

    local_loss.backward()

    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    scores = torch.zeros(N, device=device)

    w_xyz = _getparam(attr_weights, "attr_w_xyz", 1.0)
    w_feat = _getparam(attr_weights, "attr_w_feat", 0.5)
    w_opa = _getparam(attr_weights, "attr_w_opacity", 0.3)
    w_scl = _getparam(attr_weights, "attr_w_scale", 0.3)
    w_rot = _getparam(attr_weights, "attr_w_rotation", 0.2)

    if gaussians._xyz.grad is not None and w_xyz > 0:
        scores += w_xyz * gaussians._xyz.grad.norm(dim=1)
    if gaussians._features_dc.grad is not None and w_feat > 0:
        scores += w_feat * gaussians._features_dc.grad.flatten(1).norm(dim=1)
    if gaussians._features_rest.grad is not None and w_feat > 0:
        scores += (w_feat * 0.5) * gaussians._features_rest.grad.flatten(1).norm(dim=1)
    if gaussians._opacity.grad is not None and w_opa > 0:
        scores += w_opa * gaussians._opacity.grad.abs().squeeze()
    if gaussians._scaling.grad is not None and w_scl > 0:
        scores += w_scl * gaussians._scaling.grad.norm(dim=1)
    if gaussians._rotation.grad is not None and w_rot > 0:
        scores += w_rot * gaussians._rotation.grad.norm(dim=1)

    # Clean up gradients
    if hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
        gaussians.optimizer.zero_grad(set_to_none=True)
    else:
        for p in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                   gaussians._opacity, gaussians._scaling, gaussians._rotation]:
            if p.grad is not None:
                p.grad = None

    return scores.detach()


# ===========================================================================
# Multi-strategy fusion and multi-view aggregation
# ===========================================================================

def multiview_fusion(
    per_view_scores: Dict[int, Dict[str, torch.Tensor]],
    strategy_weights: Dict[str, float],
    min_views: int = 2,
    score_percentile: float = 80.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse scores from multiple strategies and views.

    Args:
        per_view_scores: {view_idx: {"ray": [N], "depth": [N], "contrib": [N], "grad": [N]}}
        strategy_weights: {"ray": w1, "depth": w2, "contrib": w3, "grad": w4}
        min_views: Minimum views a Gaussian must appear in
        score_percentile: Percentile threshold on final scores

    Returns:
        fused_scores: [N] aggregated scores
        selected_mask: [N] bool
    """
    if not per_view_scores:
        raise ValueError("No per-view scores provided")

    sample = list(per_view_scores.values())[0]
    sample_scores = list(sample.values())[0]
    N = sample_scores.shape[0]
    device = sample_scores.device

    total_score = torch.zeros(N, device=device)
    view_count = torch.zeros(N, device=device)

    for vidx, strategies in per_view_scores.items():
        # Fuse strategies for this view
        view_score = torch.zeros(N, device=device)
        total_w = 0.0
        for sname, sscores in strategies.items():
            w = strategy_weights.get(sname, 0.0)
            if w > 0:
                # Normalize scores to [0, 1] within this view
                smax = sscores.max()
                if smax > 0:
                    sscores_norm = sscores / smax
                else:
                    sscores_norm = sscores
                view_score += w * sscores_norm
                total_w += w

        if total_w > 0:
            view_score = view_score / total_w

        # Mark Gaussians with nonzero score as "seen" in this view
        seen = view_score > 1e-8
        total_score += view_score
        view_count += seen.float()

    # Average
    avg_score = torch.where(view_count > 0, total_score / view_count, torch.zeros_like(total_score))

    # Multi-view voting
    view_filter = view_count >= min_views

    voted_scores = avg_score[view_filter]
    if len(voted_scores) > 0 and score_percentile > 0:
        thresh = torch.quantile(voted_scores, score_percentile / 100.0).item()
    else:
        thresh = 0.0

    selected_mask = view_filter & (avg_score >= thresh)

    return avg_score, selected_mask


def cluster_and_expand(
    gaussians,
    selected_mask: torch.Tensor,
    scene_extent: float,
    cluster_eps: float = 0.05,
    cluster_min_samples: int = 5,
    context_expand_ratio: float = 0.1,
    remove_isolated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster selected Gaussians in 3D space and expand with context ring.

    Returns:
        target_mask, context_mask, protect_mask: [N] bool tensors
    """
    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device

    selected_indices = selected_mask.nonzero(as_tuple=True)[0]
    if len(selected_indices) == 0:
        target_mask = torch.zeros(N, dtype=torch.bool, device=device)
        context_mask = torch.zeros(N, dtype=torch.bool, device=device)
        protect_mask = torch.ones(N, dtype=torch.bool, device=device)
        return target_mask, context_mask, protect_mask

    # Only DBSCAN on the selected subset
    sel_idx_cpu = selected_indices.cpu().numpy()
    xyz_all = gaussians.get_xyz.detach()
    selected_xyz_np = xyz_all[selected_indices].cpu().numpy()

    eps_abs = cluster_eps * scene_extent
    clustering = DBSCAN(eps=eps_abs, min_samples=cluster_min_samples).fit(selected_xyz_np)
    labels = clustering.labels_

    # Build target mask
    keep = torch.from_numpy(labels != -1 if remove_isolated else np.ones(len(labels), dtype=bool))
    target_global_idx = selected_indices[keep.to(device)]

    target_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if len(target_global_idx) > 0:
        target_mask[target_global_idx] = True

    # Context ring via chunked distance computation
    context_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if target_mask.any():
        target_xyz = xyz_all[target_mask]  # [T, 3]
        T = target_xyz.shape[0]
        expand_dist = context_expand_ratio * scene_extent

        # Dynamic chunk size: each row of cdist is T*4 bytes
        bytes_per_row = T * 4
        free_mem = torch.cuda.mem_get_info()[0] if hasattr(torch.cuda, 'mem_get_info') else 4e9
        chunk_size = max(64, min(8192, int(free_mem * 0.3 / max(bytes_per_row, 1))))

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = xyz_all[start:end]
            dists = torch.cdist(chunk.unsqueeze(0), target_xyz.unsqueeze(0)).squeeze(0)
            min_dists = dists.min(dim=1).values
            context_mask[start:end] |= (min_dists <= expand_dist)
            del dists

        context_mask = context_mask & (~target_mask)

    protect_mask = ~(target_mask | context_mask)
    return target_mask, context_mask, protect_mask


# ===========================================================================
# Main entry point
# ===========================================================================

def run_full_localization(
    views: list,
    gaussians,
    render_fn,
    pipe,
    background: torch.Tensor,
    analysis_results: Dict[int, dict],
    defect_view_indices: List[int],
    scene_extent: float,
    loc_params,
    ablation_params=None,
    separate_sh: bool = False,
    use_trained_exp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full multi-strategy localization pipeline.

    Args:
        loc_params: GroupParams or dict with LocalizationParams fields
        ablation_params: GroupParams or dict with AblationParams fields

    Returns:
        target_mask, context_mask, protect_mask: [N] bool tensors
        fused_scores: [N] float tensor
    """
    # Strategy enable flags
    enable_ray = _getparam(ablation_params, "enable_ray_triangulation", True) if ablation_params else True
    enable_depth = _getparam(ablation_params, "enable_depth_backproject", True) if ablation_params else True
    enable_contrib = _getparam(ablation_params, "enable_contribution_stat", True) if ablation_params else True
    enable_grad = _getparam(ablation_params, "enable_gradient_attr", True) if ablation_params else True

    # Strategy fusion weights
    strategy_weights = {
        "ray": _getparam(loc_params, "w_strategy_ray", 1.0) if enable_ray else 0.0,
        "depth": _getparam(loc_params, "w_strategy_depth", 1.0) if enable_depth else 0.0,
        "contrib": _getparam(loc_params, "w_strategy_contrib", 0.5) if enable_contrib else 0.0,
        "grad": _getparam(loc_params, "w_strategy_grad", 1.0) if enable_grad else 0.0,
    }

    N = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    per_view_scores = {}

    num_views = len(defect_view_indices)
    for vi, vidx in enumerate(defect_view_indices):
        print(f"    Localization view {vi+1}/{num_views} (idx={vidx})", end="", flush=True)
        result = analysis_results[vidx]
        view = views[vidx]
        defect_mask = result["defect_mask"]
        if defect_mask.device.type == 'cpu':
            defect_mask = defect_mask.cuda()
        render_pkg = result["render_pkg"]
        defect_regions = result.get("defect_regions", [])

        view_strategies = {}

        with torch.no_grad():
            # --- Strategy 1: Ray triangulation ---
            if enable_ray:
                ray_scores = torch.zeros(N, device=device)
                paired = find_paired_views(
                    views, vidx,
                    angle_min=_getparam(loc_params, "ray_pair_angle_min", 15.0),
                    angle_max=_getparam(loc_params, "ray_pair_angle_max", 60.0),
                    max_pairs=_getparam(loc_params, "ray_max_pairs", 5),
                )
                for pidx in paired:
                    p_result = analysis_results.get(pidx)
                    if p_result is None:
                        continue
                    p_regions = p_result.get("defect_regions", [])
                    zones = triangulate_3d_zones(
                        view, views[pidx], defect_regions, p_regions,
                        defect_mask, p_result["defect_mask"])
                    if zones:
                        zone_scores = find_gaussians_in_zones(gaussians, zones)
                        ray_scores = torch.max(ray_scores, zone_scores)
                view_strategies["ray"] = ray_scores

            # --- Strategy 2: Depth backprojection ---
            if enable_depth:
                depth_img = render_pkg.get("depth", None)
                if depth_img is not None and depth_img.device.type == 'cpu':
                    depth_img = depth_img.cuda()
                depth_scores = depth_backproject_to_gaussians(
                    view, defect_mask, depth_img, gaussians,
                    knn_k=_getparam(loc_params, "depth_knn_k", 3),
                    local_radius=_getparam(loc_params, "depth_local_radius", 3.0),
                )
                view_strategies["depth"] = depth_scores

            # --- Strategy 3: Contribution statistics ---
            if enable_contrib:
                # Ensure radii on correct device for contribution scoring
                gpu_pkg = {}
                for pk, pv in render_pkg.items():
                    gpu_pkg[pk] = pv.cuda() if isinstance(pv, torch.Tensor) and pv.device.type == 'cpu' else pv
                contrib_scores = compute_contribution_scores(
                    gaussians, view, defect_mask, gpu_pkg,
                    min_overlap=_getparam(loc_params, "contribution_min_overlap", 0.1),
                )
                view_strategies["contrib"] = contrib_scores

        # --- Strategy 4: Gradient attribution (needs gradients, may OOM) ---
        if enable_grad:
            try:
                grad_scores = compute_gradient_attribution(
                    gaussians, view, defect_mask, render_fn, pipe, background,
                    attr_weights=loc_params,
                    separate_sh=separate_sh,
                    use_trained_exp=use_trained_exp,
                )
                view_strategies["grad"] = grad_scores
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    if vi == 0:
                        print(" [WARN: gradient attribution OOM, disabling for remaining views]", end="")
                    enable_grad = False
                    torch.cuda.empty_cache()
                else:
                    raise

        strategies_used = list(view_strategies.keys())
        print(f" -> {strategies_used}")
        per_view_scores[vidx] = view_strategies

    # Multi-strategy + multi-view fusion
    if not per_view_scores:
        target_mask = torch.zeros(N, dtype=torch.bool, device=device)
        context_mask = torch.zeros(N, dtype=torch.bool, device=device)
        protect_mask = torch.ones(N, dtype=torch.bool, device=device)
        return target_mask, context_mask, protect_mask, torch.zeros(N, device=device)

    fused_scores, selected_mask = multiview_fusion(
        per_view_scores, strategy_weights,
        min_views=_getparam(loc_params, "vote_min_views", 2),
        score_percentile=_getparam(loc_params, "vote_score_percentile", 80.0),
    )

    # 3D clustering & expansion
    target_mask, context_mask, protect_mask = cluster_and_expand(
        gaussians, selected_mask, scene_extent,
        cluster_eps=_getparam(loc_params, "cluster_eps", 0.05),
        cluster_min_samples=_getparam(loc_params, "cluster_min_samples", 5),
        context_expand_ratio=_getparam(loc_params, "context_expand_ratio", 0.1),
        remove_isolated=_getparam(loc_params, "remove_isolated", True),
    )

    return target_mask, context_mask, protect_mask, fused_scores
