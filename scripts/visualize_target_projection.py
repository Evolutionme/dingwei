#!/usr/bin/env python3
"""
Target Projection Visualizer
=============================
Overlay target Gaussians onto defect views for debugging localization quality.

Output per view:
  - Red    : defect mask (from error analysis)
  - Green  : target Gaussian projection footprint
  - Yellow : overlap (target correctly covers defect)

Usage:
    python scripts/visualize_target_projection.py \
        -m /path/to/output \
        -s /path/to/dataset \
        --iteration 30000 \
        --refine_tag exp1 \
        [--max_views 10] \
        [--point_radius 2] \
        [--opacity 0.5]
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser


def project_points_to_image(xyz, camera):
    """Project 3D Gaussian centers onto the 2D image plane of a camera.

    Args:
        xyz: (N, 3) tensor of 3D positions
        camera: Camera object with full_proj_transform, image_width, image_height

    Returns:
        px, py: (N,) integer pixel coordinates
        valid: (N,) boolean mask for points inside the image
    """
    N = xyz.shape[0]
    device = camera.full_proj_transform.device

    # Homogeneous coordinates
    ones = torch.ones(N, 1, device=device)
    xyz_h = torch.cat([xyz.to(device), ones], dim=1)  # (N, 4)

    # Project: clip space
    proj = xyz_h @ camera.full_proj_transform  # (N, 4)
    w = proj[:, 3:4].clamp(min=1e-6)
    ndc = proj[:, :3] / w  # (N, 3) in [-1, 1]

    # NDC -> pixel
    W = camera.image_width
    H = camera.image_height
    px = ((ndc[:, 0] + 1.0) * 0.5 * W).long()
    py = ((ndc[:, 1] + 1.0) * 0.5 * H).long()

    # Validity: in front of camera and within image bounds
    valid = (w.squeeze() > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    return px, py, valid


def make_overlay(gt_image, defect_mask, target_proj_mask, opacity=0.5):
    """Compose the RGB overlay image.

    Args:
        gt_image: (3, H, W) tensor, float [0, 1]
        defect_mask: (H, W) boolean tensor
        target_proj_mask: (H, W) boolean tensor
        opacity: overlay blending factor

    Returns:
        overlay: (H, W, 3) numpy uint8 image
    """
    H, W = defect_mask.shape
    base = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

    # Color layers
    red = np.array([255, 60, 60], dtype=np.uint8)    # defect only
    green = np.array([60, 255, 60], dtype=np.uint8)   # target only
    yellow = np.array([255, 255, 60], dtype=np.uint8)  # overlap

    d = defect_mask.cpu().numpy()
    t = target_proj_mask.cpu().numpy()

    overlap = d & t
    defect_only = d & ~t
    target_only = t & ~d

    for mask, color in [(defect_only, red), (target_only, green), (overlap, yellow)]:
        region = mask.astype(bool)
        base[region] = (
            base[region].astype(np.float32) * (1 - opacity)
            + color.astype(np.float32) * opacity
        ).astype(np.uint8)

    return base


def compute_stats(defect_mask, target_proj_mask):
    """Compute overlap statistics."""
    d = defect_mask.bool()
    t = target_proj_mask.bool()

    d_area = d.sum().item()
    t_area = t.sum().item()
    overlap_area = (d & t).sum().item()
    defect_miss = (d & ~t).sum().item()
    target_outside = (t & ~d).sum().item()

    recall = overlap_area / max(d_area, 1)       # defect covered by target
    precision = overlap_area / max(t_area, 1)     # target that hits defect

    return {
        "defect_px": d_area,
        "target_px": t_area,
        "overlap_px": overlap_area,
        "defect_miss_px": defect_miss,
        "target_outside_px": target_outside,
        "recall": recall,
        "precision": precision,
    }


def main():
    parser = ArgumentParser(description="Visualize target Gaussian projections on defect views")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=30000, type=int,
                        help="Model iteration to load")
    parser.add_argument("--refine_tag", default="exp1", type=str,
                        help="Refine tag for loading regions and analysis")
    parser.add_argument("--max_views", default=0, type=int,
                        help="Max views to visualize (0 = all)")
    parser.add_argument("--point_radius", default=2, type=int,
                        help="Pixel radius for each projected Gaussian point")
    parser.add_argument("--opacity", default=0.5, type=float,
                        help="Overlay opacity [0-1]")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    dataset = model.extract(args)

    output_dir = dataset.model_path
    tag = args.refine_tag
    iteration = args.iteration
    max_views = args.max_views
    radius = args.point_radius
    opacity = args.opacity

    # ── Output directory ──
    vis_dir = os.path.join(output_dir, "visualizations", tag, "target_projection")
    os.makedirs(vis_dir, exist_ok=True)

    # ── Load model ──
    print(f"Loading model from {output_dir} at iteration {iteration}")
    original_data_device = dataset.data_device
    dataset.data_device = "cpu"
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    dataset.data_device = original_data_device

    xyz = gaussians.get_xyz.detach()  # (N, 3)
    print(f"  Total Gaussians: {xyz.shape[0]}")

    # ── Load target mask ──
    mask_path = os.path.join(output_dir, "regions", tag, "target_mask.pt")
    if not os.path.exists(mask_path):
        print(f"ERROR: target_mask.pt not found at {mask_path}")
        print("  Run refine.py (with analysis) first to generate regions.")
        sys.exit(1)
    target_mask = torch.load(mask_path, map_location="cpu")
    print(f"  Target Gaussians: {target_mask.sum().item()} / {target_mask.shape[0]}")

    target_xyz = xyz[target_mask]  # (T, 3)

    # ── Load defect view indices ──
    dv_path = os.path.join(output_dir, "analysis", tag, "defect_views.json")
    if not os.path.exists(dv_path):
        print(f"ERROR: defect_views.json not found at {dv_path}")
        sys.exit(1)
    with open(dv_path, "r") as f:
        defect_view_indices = json.load(f)

    # ── Load defect masks directory ──
    dm_dir = os.path.join(output_dir, "analysis", tag, "defect_masks")

    # ── Get train cameras ──
    train_cams = scene.getTrainCameras()
    print(f"  Train cameras: {len(train_cams)}")
    print(f"  Defect views: {len(defect_view_indices)}")

    if max_views > 0:
        defect_view_indices = defect_view_indices[:max_views]
        print(f"  Visualizing first {max_views} views")

    # ── Process each defect view ──
    all_stats = []
    print(f"\nGenerating overlays → {vis_dir}/")
    print("-" * 70)

    for i, vidx in enumerate(defect_view_indices):
        cam = train_cams[vidx]
        H, W = cam.image_height, cam.image_width

        # Project target Gaussians
        px, py, valid = project_points_to_image(target_xyz, cam)
        px, py = px[valid], py[valid]

        # Build target projection mask with radius
        target_proj = torch.zeros(H, W, dtype=torch.bool)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    cx = (px + dx).clamp(0, W - 1)
                    cy = (py + dy).clamp(0, H - 1)
                    target_proj[cy, cx] = True

        # Load defect mask
        dm_path = os.path.join(dm_dir, f"{vidx:05d}.png")
        if os.path.exists(dm_path):
            dm_img = np.array(Image.open(dm_path).convert("L").resize((W, H), Image.NEAREST))
            defect_mask = torch.from_numpy(dm_img > 127)
        else:
            # Fallback: no defect mask available
            defect_mask = torch.zeros(H, W, dtype=torch.bool)

        # Compute stats
        stats = compute_stats(defect_mask, target_proj)
        stats["view_idx"] = vidx
        all_stats.append(stats)

        # Make overlay
        gt_image = cam.original_image[:3, :, :]
        if gt_image.device != torch.device("cpu"):
            gt_image = gt_image.cpu()
        overlay = make_overlay(gt_image, defect_mask, target_proj, opacity=opacity)

        # Save
        out_path = os.path.join(vis_dir, f"view_{vidx:05d}.png")
        Image.fromarray(overlay).save(out_path)

        status = "✓" if stats["recall"] > 0.5 else "✗"
        print(f"  [{status}] View {vidx:4d}  |  "
              f"recall={stats['recall']:.1%}  precision={stats['precision']:.1%}  |  "
              f"defect={stats['defect_px']:6d}px  target={stats['target_px']:6d}px  "
              f"overlap={stats['overlap_px']:6d}px")

    # ── Summary ──
    print("-" * 70)
    if all_stats:
        avg_recall = np.mean([s["recall"] for s in all_stats])
        avg_precision = np.mean([s["precision"] for s in all_stats])
        total_defect = sum(s["defect_px"] for s in all_stats)
        total_overlap = sum(s["overlap_px"] for s in all_stats)
        total_miss = sum(s["defect_miss_px"] for s in all_stats)

        print(f"\n  Summary ({len(all_stats)} views):")
        print(f"    Avg recall    : {avg_recall:.1%}  (defect area covered by target)")
        print(f"    Avg precision : {avg_precision:.1%}  (target area that hits defect)")
        print(f"    Total defect  : {total_defect:,} px")
        print(f"    Total overlap : {total_overlap:,} px")
        print(f"    Total miss    : {total_miss:,} px")

        # Save stats JSON
        stats_path = os.path.join(vis_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump({"per_view": all_stats, "summary": {
                "avg_recall": avg_recall,
                "avg_precision": avg_precision,
                "total_defect_px": total_defect,
                "total_overlap_px": total_overlap,
                "total_miss_px": total_miss,
                "num_views": len(all_stats),
            }}, f, indent=2)
        print(f"\n  Stats saved to {stats_path}")

    print(f"  Overlays saved to {vis_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
