#
# Error Analysis Module for Local Refinement (v2)
# Constructs per-view error maps, extracts defect masks, and performs
# multi-view defect association.
#
# v2 changes:
#   - Fix: access GroupParams via getattr() instead of .get()
#   - Add: luminance normalization option (from GS-LPM)
#   - Add: compute_adaptive_error_map() patch-based adaptive threshold (from GS-LPM)
#   - Add: extract_defect_regions() connected-component bboxes (from GS-LPM)
#   - Add: normalize composite error map to [0, 1]
#

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


def _getparam(params, name, default):
    """Safely get a parameter from either a GroupParams object or a dict."""
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def compute_rgb_error(render: torch.Tensor, gt: torch.Tensor,
                      normalize_luminance: bool = False) -> torch.Tensor:
    """Per-pixel L1 photometric error. Shape: [H, W].

    Args:
        render: Rendered image [C, H, W]
        gt: Ground truth image [C, H, W]
        normalize_luminance: If True, normalize both images by their mean
            luminance before computing error (from GS-LPM get_errormap).
    """
    if normalize_luminance:
        r_mean = render.mean()
        g_mean = gt.mean()
        if r_mean > 1e-6 and g_mean > 1e-6:
            render = render * (g_mean / r_mean)
    return torch.abs(render - gt).mean(dim=0)


def compute_ssim_error_map(render: torch.Tensor, gt: torch.Tensor,
                           window_size: int = 11) -> torch.Tensor:
    """Per-pixel structural dissimilarity (1 - SSIM). Shape: [H, W].
    Operates on [C, H, W] tensors."""
    C = render.shape[0]
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=render.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    img1 = render.unsqueeze(0)  # [1, C, H, W]
    img2 = gt.unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 1 - SSIM gives dissimilarity; average over channels
    dssim = (1.0 - ssim_map.squeeze(0)).mean(dim=0)
    return dssim.clamp(0.0, 1.0)


def compute_edge_error(render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-pixel edge gradient difference. Shape: [H, W].
    Uses Sobel filters to capture edge discrepancies."""
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=render.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=render.device).unsqueeze(0).unsqueeze(0)

    # Convert to grayscale
    render_gray = render.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    gt_gray = gt.mean(dim=0, keepdim=True).unsqueeze(0)

    # Compute edges
    render_edge_x = F.conv2d(render_gray, sobel_x, padding=1)
    render_edge_y = F.conv2d(render_gray, sobel_y, padding=1)
    render_edge = torch.sqrt(render_edge_x ** 2 + render_edge_y ** 2 + 1e-8)

    gt_edge_x = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_edge_y = F.conv2d(gt_gray, sobel_y, padding=1)
    gt_edge = torch.sqrt(gt_edge_x ** 2 + gt_edge_y ** 2 + 1e-8)

    edge_diff = torch.abs(render_edge - gt_edge).squeeze(0).squeeze(0)
    return edge_diff


def compute_depth_error(render_depth: torch.Tensor,
                        gt_depth: torch.Tensor,
                        depth_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-pixel depth error. Shape: [H, W]."""
    diff = torch.abs(render_depth - gt_depth)
    if depth_mask is not None:
        diff = diff * depth_mask
    if diff.dim() == 3:
        diff = diff.squeeze(0)
    return diff


def compute_composite_error_map(
    render: torch.Tensor,
    gt: torch.Tensor,
    w_rgb: float = 1.0,
    w_ssim: float = 0.5,
    w_edge: float = 0.3,
    w_depth: float = 0.0,
    w_lpips: float = 0.0,
    render_depth: Optional[torch.Tensor] = None,
    gt_depth: Optional[torch.Tensor] = None,
    depth_mask: Optional[torch.Tensor] = None,
    normalize_luminance: bool = False,
) -> torch.Tensor:
    """Construct a composite per-pixel error map from multiple error terms.

    Args:
        render: Rendered image [C, H, W]
        gt: Ground truth image [C, H, W]
        w_*: Weights for each error component (set to 0 to disable)
        render_depth: Rendered depth map [1, H, W] or [H, W]
        gt_depth: GT depth map [1, H, W] or [H, W]
        depth_mask: Valid depth mask [1, H, W] or [H, W]
        normalize_luminance: Normalize luminance before L1 (from GS-LPM)

    Returns:
        Composite error map [H, W], normalized to [0, 1]
    """
    error_map = torch.zeros(render.shape[1], render.shape[2], device=render.device)
    total_w = 0.0

    if w_rgb > 0:
        e_rgb = compute_rgb_error(render, gt, normalize_luminance=normalize_luminance)
        error_map += w_rgb * e_rgb
        total_w += w_rgb

    if w_ssim > 0:
        e_ssim = compute_ssim_error_map(render, gt)
        error_map += w_ssim * e_ssim
        total_w += w_ssim

    if w_edge > 0:
        e_edge = compute_edge_error(render, gt)
        error_map += w_edge * e_edge
        total_w += w_edge

    if w_depth > 0 and render_depth is not None and gt_depth is not None:
        e_depth = compute_depth_error(render_depth, gt_depth, depth_mask)
        error_map += w_depth * e_depth
        total_w += w_depth

    # Normalize by total weight
    if total_w > 0:
        error_map = error_map / total_w

    # Clamp to [0, 1] for consistency
    emax = error_map.max()
    if emax > 0:
        error_map = error_map / emax

    return error_map


def compute_adaptive_error_map(
    error_map: torch.Tensor,
    patch_size: int = 16,
    fill_ratio: float = 0.5,
    global_percentile: float = 90.0,
) -> torch.Tensor:
    """Patch-based adaptive thresholding (from GS-LPM get_errormap).

    Divides the error map into patches. For each patch, if a sufficient
    fraction (fill_ratio) of pixels exceed a local threshold (derived from
    the patch mean), the entire patch is marked as defective.

    Args:
        error_map: Per-pixel error [H, W] (already in [0, 1])
        patch_size: Size of square patch
        fill_ratio: Minimum fraction of above-threshold pixels within a patch
        global_percentile: Fallback global percentile for sparsely-defective patches

    Returns:
        Adaptive binary mask [H, W] (torch.bool)
    """
    device = error_map.device
    H, W = error_map.shape
    em_np = error_map.detach().cpu().numpy()
    mask = np.zeros((H, W), dtype=np.uint8)

    global_thresh = np.percentile(em_np, global_percentile)

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patch = em_np[y:y_end, x:x_end]

            # Local threshold: use patch mean as reference
            local_thresh = max(patch.mean() * 1.5, global_thresh)
            above = (patch >= local_thresh).astype(np.float32)
            ratio = above.mean()

            if ratio >= fill_ratio:
                mask[y:y_end, x:x_end] = 1

    return torch.from_numpy(mask.astype(bool)).to(device)


def extract_defect_mask(
    error_map: torch.Tensor,
    percentile: float = 90.0,
    abs_threshold: float = 0.0,
    min_area: int = 64,
    dilate_radius: int = 5,
    use_adaptive: bool = False,
    adaptive_patch_size: int = 16,
    adaptive_fill_ratio: float = 0.5,
) -> torch.Tensor:
    """Extract binary defect mask from an error map.

    Args:
        error_map: Per-pixel error [H, W]
        percentile: Top-k percentile threshold
        abs_threshold: Absolute error threshold (used if > 0)
        min_area: Minimum connected component area
        dilate_radius: Morphological dilation kernel radius
        use_adaptive: Use adaptive patch-based thresholding (from GS-LPM)
        adaptive_patch_size: Patch size for adaptive threshold
        adaptive_fill_ratio: Fill ratio for adaptive threshold

    Returns:
        Binary defect mask [H, W] (torch.bool on same device)
    """
    device = error_map.device

    if use_adaptive:
        # Use GS-LPM style adaptive thresholding
        adaptive_mask = compute_adaptive_error_map(
            error_map, adaptive_patch_size, adaptive_fill_ratio, percentile)
        # Also apply standard percentile mask and union them
        em_np = error_map.detach().cpu().numpy()
        thresh = np.percentile(em_np, percentile)
        pct_mask = (em_np >= thresh)
        mask = (adaptive_mask.cpu().numpy() | pct_mask).astype(np.uint8)
    else:
        em_np = error_map.detach().cpu().numpy()
        if abs_threshold > 0:
            thresh = abs_threshold
        else:
            thresh = np.percentile(em_np, percentile)
        mask = (em_np >= thresh).astype(np.uint8)

    # Morphological dilation
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Connected components: remove small regions
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 1
        mask = clean_mask

    return torch.from_numpy(mask.astype(bool)).to(device)


def extract_defect_regions(
    defect_mask: torch.Tensor,
    min_area: int = 64,
) -> List[Dict]:
    """Extract connected-component bounding boxes from a defect mask.
    (From GS-LPM region_matching: find connected components and their bboxes.)

    Args:
        defect_mask: Binary defect mask [H, W] (bool)
        min_area: Minimum area to keep

    Returns:
        List of dicts, each with keys:
            "bbox": (x, y, w, h) — bounding box in pixel coords
            "area": int — component area
            "centroid": (cx, cy) — centroid in pixel coords
    """
    mask_np = defect_mask.detach().cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_np, connectivity=8)

    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        regions.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "centroid": (float(cx), float(cy)),
        })

    return regions


def analyze_all_views(
    views: list,
    gaussians,
    render_fn,
    pipe,
    background: torch.Tensor,
    error_params,
    separate_sh: bool = False,
    use_trained_exp: bool = False,
) -> Dict[int, dict]:
    """Run error analysis on all views, producing per-view defect masks.

    Args:
        views: List of Camera objects
        gaussians: GaussianModel instance
        render_fn: The render function from gaussian_renderer
        pipe: PipelineParams
        background: Background tensor
        error_params: GroupParams object or dict with ErrorAnalysisParams fields
        separate_sh: Whether to use separate SH
        use_trained_exp: Whether to use trained exposure

    Returns:
        Dict mapping view index -> {
            "error_map": [H, W] tensor,
            "defect_mask": [H, W] bool tensor,
            "defect_regions": list of region dicts,
            "render": [C, H, W] tensor,
            "gt": [C, H, W] tensor,
            "render_pkg": dict from renderer
        }
    """
    results = {}

    # Extract params via _getparam (works with both GroupParams and dict)
    w_rgb = _getparam(error_params, "w_rgb", 1.0)
    w_ssim = _getparam(error_params, "w_ssim", 0.5)
    w_edge = _getparam(error_params, "w_edge", 0.3)
    w_depth = _getparam(error_params, "w_depth", 0.0)
    w_lpips = _getparam(error_params, "w_lpips", 0.0)
    percentile = _getparam(error_params, "error_percentile", 90.0)
    abs_threshold = _getparam(error_params, "error_abs_threshold", 0.0)
    min_area = _getparam(error_params, "min_defect_area", 64)
    dilate_radius = _getparam(error_params, "mask_dilate_radius", 5)
    use_adaptive = _getparam(error_params, "use_adaptive_threshold", False)
    adaptive_patch_size = _getparam(error_params, "adaptive_patch_size", 16)
    adaptive_fill_ratio = _getparam(error_params, "adaptive_fill_ratio", 0.5)

    for idx, view in enumerate(views):
        # No gradients needed for error analysis — avoids rasterizer buffer
        # overflow with large Gaussian counts (autograd buffers scale with N).
        with torch.no_grad():
            render_pkg = render_fn(
                view, gaussians, pipe, background,
                use_trained_exp=use_trained_exp,
                separate_sh=separate_sh
            )
            rendered = render_pkg["render"]
            gt = view.original_image[:3, :, :].cuda()

            # Handle alpha mask
            if view.alpha_mask is not None:
                alpha_mask = view.alpha_mask.cuda()
                rendered = rendered * alpha_mask

            # Depth data
            render_depth = render_pkg.get("depth", None)
            gt_depth = None
            depth_mask = None
            if hasattr(view, 'invdepthmap') and view.invdepthmap is not None:
                gt_depth = view.invdepthmap.cuda()
                if hasattr(view, 'depth_mask'):
                    depth_mask = view.depth_mask.cuda()

            # Composite error map
            error_map = compute_composite_error_map(
                rendered, gt,
                w_rgb=w_rgb, w_ssim=w_ssim, w_edge=w_edge,
                w_depth=w_depth, w_lpips=w_lpips,
                render_depth=render_depth,
                gt_depth=gt_depth,
                depth_mask=depth_mask,
                normalize_luminance=True,
            )

            # Extract defect mask
            defect_mask = extract_defect_mask(
                error_map,
                percentile=percentile,
                abs_threshold=abs_threshold,
                min_area=min_area,
                dilate_radius=dilate_radius,
                use_adaptive=use_adaptive,
                adaptive_patch_size=adaptive_patch_size,
                adaptive_fill_ratio=adaptive_fill_ratio,
            )

            # Extract defect regions (bounding boxes)
            defect_regions = extract_defect_regions(defect_mask, min_area=min_area)

        # Only keep fields needed downstream (localization uses depth + radii).
        # Do NOT store full render_pkg — viewspace_points [N,3] per view
        # would accumulate ~12GB+ across all views.
        slim_pkg = {}
        if "depth" in render_pkg and render_pkg["depth"] is not None:
            slim_pkg["depth"] = render_pkg["depth"].detach().cpu()
        if "radii" in render_pkg and render_pkg["radii"] is not None:
            slim_pkg["radii"] = render_pkg["radii"].detach().cpu()

        results[idx] = {
            "error_map": error_map.cpu(),
            "defect_mask": defect_mask,  # bool, small
            "defect_regions": defect_regions,
            "render_pkg": slim_pkg,
        }

    return results


def filter_views_with_defects(
    analysis_results: Dict[int, dict],
    min_defect_ratio: float = 0.001,
) -> List[int]:
    """Return indices of views that have significant defect regions.

    Args:
        analysis_results: Output of analyze_all_views
        min_defect_ratio: Minimum ratio of defect pixels to total pixels

    Returns:
        List of view indices with significant defects
    """
    defect_views = []
    for idx, result in analysis_results.items():
        mask = result["defect_mask"]
        ratio = mask.float().mean().item()
        if ratio >= min_defect_ratio:
            defect_views.append(idx)
    return defect_views
