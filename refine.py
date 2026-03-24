#
# Local Refinement Pipeline for 3D Gaussian Splatting (v2)
# Main entry point: baseline loading -> analysis -> localization -> refinement -> evaluation
#
# v2 changes:
#   - Correct function signatures to match v2 modules
#   - Add AblationParams support
#   - Integrate opacity calibration, boundary loss, grad_ratio
#   - Fix exposure init, background color, GroupParams access
#   - Use save_error_analysis_results from region_utils
#

import torch
import torch.nn.functional as F
import argparse
import sys
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from arguments.refine_args import (
    ErrorAnalysisParams, LocalizationParams, RefinementParams, AblationParams
)
from scene import Scene
from scene.gaussian_model_local import LocalGaussianModel
from gaussian_renderer import render
from gaussian_renderer.render_analysis import render_with_local_loss, compute_view_defect_weight
from utils.error_analysis import analyze_all_views, filter_views_with_defects
from utils.localization import run_full_localization
from utils.region_utils import RegionManager, save_error_analysis_results
from utils.loss_utils import l1_loss, ssim as compute_ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ============================================================================
# Stage A: Load baseline model
# ============================================================================

def load_baseline(dataset, iteration, opt):
    """Load a pre-trained baseline model as LocalGaussianModel.

    Returns:
        gaussians: LocalGaussianModel with loaded weights
        scene: Scene object
    """
    # Force images to stay on CPU — with 194 cameras at 1.6K resolution,
    # keeping them all on GPU wastes ~4GB+ VRAM and causes OOM during render.
    # Images are moved to GPU on-the-fly by render() / .cuda() calls.
    original_data_device = dataset.data_device
    dataset.data_device = "cpu"

    gaussians = LocalGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    dataset.data_device = original_data_device  # restore

    # Ensure _exposure is initialized (load_ply path doesn't do this)
    if not hasattr(gaussians, '_exposure') or gaussians._exposure is None:
        train_cams = scene.getTrainCameras()
        gaussians.exposure_mapping = {
            cam.image_name: idx for idx, cam in enumerate(train_cams)
        }
        gaussians.pretrained_exposures = None

        exposure_file = os.path.join(dataset.model_path, "exposure.json")
        if os.path.exists(exposure_file):
            try:
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                exposure_list = []
                for cam in train_cams:
                    if cam.image_name in exposures:
                        exposure_list.append(
                            torch.FloatTensor(exposures[cam.image_name]).cuda())
                    else:
                        exposure_list.append(torch.eye(3, 4, device="cuda"))
                exposure = torch.stack(exposure_list, dim=0)
                print(f"  Loaded exposure from {exposure_file}")
            except Exception as e:
                print(f"  Warning: Failed to load exposure.json: {e}")
                exposure = torch.eye(3, 4, device="cuda")[None].repeat(
                    len(train_cams), 1, 1)
        else:
            exposure = torch.eye(3, 4, device="cuda")[None].repeat(
                len(train_cams), 1, 1)

        gaussians._exposure = torch.nn.Parameter(
            exposure.requires_grad_(True))

    # NOTE: Do NOT call training_setup() here. Optimizer states (~1.8GB for
    # 3.9M Gaussians) are not needed during analysis and would cause OOM.
    # training_setup() is called later in run_local_refinement().
    return gaussians, scene


# ============================================================================
# Stage B: Error analysis & localization
# ============================================================================

def run_analysis_and_localization(
    gaussians, scene, pipe, background,
    error_params, loc_params, ablation_params,
    output_dir, refine_tag,
    separate_sh: bool = True,
    max_loc_views: int = 0,
):
    """Run error analysis and multi-strategy 2D-3D localization.

    Returns:
        region_manager: RegionManager with target/context/protect masks
        analysis_results: dict of per-view analysis data
        defect_view_indices: list of view indices with defects
    """
    print("\n=== Stage B: Error Analysis & Localization ===")

    # B1: Analyze all training views
    print("  [B1] Computing error maps...")
    train_cams = scene.getTrainCameras()
    analysis_results = analyze_all_views(
        train_cams, gaussians, render, pipe, background,
        error_params,
        separate_sh=separate_sh,
        use_trained_exp=False,
    )

    # Save analysis outputs
    analysis_dir = os.path.join(output_dir, "analysis", refine_tag)
    save_error_analysis_results(analysis_results, analysis_dir)

    # Find views with defects
    defect_view_indices = filter_views_with_defects(analysis_results, min_defect_ratio=0.001)

    with open(os.path.join(analysis_dir, "defect_views.json"), "w") as f:
        json.dump(defect_view_indices, f)

    print(f"  Analysis saved to {analysis_dir}")
    print(f"  Views with defects: {len(defect_view_indices)}/{len(train_cams)}")

    # Optionally limit localization views
    if max_loc_views > 0 and len(defect_view_indices) > max_loc_views:
        print(f"  Limiting localization to {max_loc_views}/{len(defect_view_indices)} views")
        defect_view_indices = defect_view_indices[:max_loc_views]

    if len(defect_view_indices) == 0:
        print("  WARNING: No defect views found. Creating empty regions.")
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device
        region_manager = RegionManager(
            target_mask=torch.zeros(N, dtype=torch.bool, device=device),
            context_mask=torch.zeros(N, dtype=torch.bool, device=device),
            protect_mask=torch.ones(N, dtype=torch.bool, device=device),
        )
        return region_manager, analysis_results, defect_view_indices

    # B2-B4: Multi-strategy localization
    print("  [B2-B4] Running multi-strategy 2D-3D localization...")
    target_mask, context_mask, protect_mask, fused_scores = run_full_localization(
        views=list(train_cams),
        gaussians=gaussians,
        render_fn=render,
        pipe=pipe,
        background=background,
        analysis_results=analysis_results,
        defect_view_indices=defect_view_indices,
        scene_extent=scene.cameras_extent,
        loc_params=loc_params,
        ablation_params=ablation_params,
        separate_sh=separate_sh,
        use_trained_exp=False,
    )

    region_manager = RegionManager(
        target_mask=target_mask,
        context_mask=context_mask,
        protect_mask=protect_mask,
        scores=fused_scores,
    )

    print(f"  Region segmentation complete:")
    print(region_manager.summary())

    return region_manager, analysis_results, defect_view_indices


# ============================================================================
# Stage C: Local refinement
# ============================================================================

def run_local_refinement(
    gaussians, scene, pipe, background,
    region_manager, analysis_results, defect_view_indices,
    opt, refine_params,
    output_dir, refine_tag, tb_writer=None,
    separate_sh: bool = True,
):
    """Run region-aware local refinement optimization."""
    print("\n=== Stage C: Local Refinement ===")

    train_cams = scene.getTrainCameras()

    # Set regions on model
    gaussians.set_regions(
        region_manager.target_mask,
        region_manager.context_mask,
        region_manager.protect_mask,
    )

    # Snapshot baseline for anchor regularization
    gaussians.snapshot_baseline()

    # Setup region-aware optimizer (training_args first, then refine_args)
    gaussians.refine_training_setup(opt, refine_params)

    # Opacity calibration (from GS-LPM)
    if refine_params.calibrate_opacity:
        print(f"  Calibrating target opacity (top_ratio={refine_params.calibrate_top_ratio})")
        gaussians.calibrate_target_opacity(
            top_ratio=refine_params.calibrate_top_ratio,
            value=refine_params.calibrate_value,
        )

    # Build per-view defect masks dict for quick lookup
    defect_masks = {}
    for vidx in range(len(train_cams)):
        if vidx in analysis_results:
            defect_masks[vidx] = analysis_results[vidx]["defect_mask"]
        else:
            defect_masks[vidx] = None

    # Build sampling weights (defect views get more samples)
    defect_set = set(defect_view_indices)
    view_weights = np.array([
        3.0 if i in defect_set else 1.0
        for i in range(len(train_cams))
    ], dtype=np.float64)
    view_weights /= view_weights.sum()

    # Refinement loop
    n_iters = refine_params.refine_iterations
    ema_loss = 0.0
    progress_bar = tqdm(range(n_iters), desc="Refinement", leave=False)

    for iteration in progress_bar:
        # Sample view
        view_idx = int(np.random.choice(len(train_cams), p=view_weights))
        cam = train_cams[view_idx]

        # Get defect mask for this view (may be None for non-defect views)
        defect_mask = defect_masks.get(view_idx, None)

        # Render with region-aware loss
        result = render_with_local_loss(
            viewpoint_cam=cam,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            defect_mask=defect_mask,
            region_manager=region_manager,
            refine_params=refine_params,
            separate_sh=separate_sh,
            use_trained_exp=False,
        )

        loss_total = result["loss_total"]

        # Backward
        loss_total.backward()

        with torch.no_grad():
            # Apply gradient mask (region-aware protection)
            gaussians.apply_gradient_mask()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # Update learning rate
            gaussians.update_learning_rate(iteration)

            # Exposure optimizer
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            # Accumulate gradients for densification
            render_pkg = result["render_pkg"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # Local densify/prune
            if refine_params.local_densify:
                if (iteration % refine_params.local_densify_interval == 0 and
                        refine_params.local_densify_from_iter <= iteration <= refine_params.local_densify_until_iter):
                    gaussians.local_densify_and_prune(
                        max_grad=refine_params.local_densify_grad_threshold,
                        min_opacity=0.005,
                        extent=scene.cameras_extent,
                        max_screen_size=20,
                        radii=radii,
                        grad_ratio=refine_params.grad_ratio,
                        prune_hysteresis=refine_params.prune_hysteresis,
                    )

        # EMA loss tracking
        ema_loss = 0.05 * loss_total.item() + 0.95 * ema_loss

        if tb_writer and iteration % 10 == 0:
            tb_writer.add_scalar("refine/total_loss", loss_total.item(), iteration)
            tb_writer.add_scalar("refine/local_loss", result["loss_local"].item(), iteration)
            tb_writer.add_scalar("refine/anchor_loss", result["loss_anchor"].item(), iteration)
            tb_writer.add_scalar("refine/context_loss", result["loss_context"].item(), iteration)
            tb_writer.add_scalar("refine/boundary_loss", result["loss_boundary"].item(), iteration)

        progress_bar.set_postfix({"loss": f"{ema_loss:.6f}"})

    print(f"  Refinement complete. Final EMA loss: {ema_loss:.6f}")
    print(f"  Final Gaussian count: {gaussians.get_xyz.shape[0]}")


# ============================================================================
# Evaluation
# ============================================================================

def refine_eval(scene, gaussians, pipe, background, tb_writer, refine_tag, separate_sh=False):
    """Quick evaluation on test cameras after refinement."""
    torch.cuda.empty_cache()
    test_cams = scene.getTestCameras()
    if not test_cams or len(test_cams) == 0:
        print("  No test cameras available for evaluation.")
        return None, None

    print(f"\n=== Evaluation on {len(test_cams)} test cameras ===")
    total_psnr = 0.0
    total_ssim = 0.0

    for cam in test_cams:
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe, background, separate_sh=separate_sh)
            image = render_pkg["render"]
            gt = cam.original_image[:3, :, :].cuda()

            if cam.alpha_mask is not None:
                image = image * cam.alpha_mask.cuda()

            psnr_val = psnr(image, gt).mean().item()
            ssim_val = compute_ssim(image, gt)
            if isinstance(ssim_val, torch.Tensor):
                ssim_val = ssim_val.mean().item()

            total_psnr += psnr_val
            total_ssim += ssim_val

    avg_psnr = total_psnr / len(test_cams)
    avg_ssim = total_ssim / len(test_cams)

    print(f"  Test PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

    if tb_writer:
        tb_writer.add_scalar(f"refine_eval/psnr", avg_psnr, 0)
        tb_writer.add_scalar(f"refine_eval/ssim", avg_ssim, 0)

    return avg_psnr, avg_ssim


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Local Refinement for 3D Gaussian Splatting (v2)")

    # Standard 3DGS arguments
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    opt_params = OptimizationParams(parser)

    # Refinement-specific arguments
    err_params = ErrorAnalysisParams(parser)
    loc_params = LocalizationParams(parser)
    ref_params = RefinementParams(parser)
    abl_params = AblationParams(parser)

    # Additional arguments
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Baseline iteration to load (-1 = latest)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refine_tag", type=str, default="refine",
                        help="Tag for refinement run")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip analysis, load existing regions")
    parser.add_argument("--skip_refine", action="store_true",
                        help="Only run analysis, skip refinement")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation after refinement")
    parser.add_argument("--max_loc_views", type=int, default=0,
                        help="Max views for localization (0=all, useful for quick tests)")

    args = get_combined_args(parser)

    # Sanitize depths (cfg_args may have whitespace-only value)
    if hasattr(args, 'depths') and args.depths and args.depths.strip() == "":
        args.depths = ""

    # Extract parameter groups
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    opt = opt_params.extract(args)
    error_params = err_params.extract(args)
    localization_params = loc_params.extract(args)
    refine_params = ref_params.extract(args)
    ablation_params = abl_params.extract(args)

    # Background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Output directories
    output_dir = dataset.model_path
    os.makedirs(os.path.join(output_dir, "analysis", args.refine_tag), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "regions", args.refine_tag), exist_ok=True)

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(
            os.path.join(output_dir, f"refine_logs_{args.refine_tag}"))

    # Print config
    print("=" * 60)
    print("Local Refinement Pipeline (v2)")
    print("=" * 60)
    print(f"  Model path:    {dataset.model_path}")
    print(f"  Source path:   {dataset.source_path}")
    print(f"  Iteration:     {args.iteration}")
    print(f"  Refine iters:  {refine_params.refine_iterations}")
    print(f"  Protect mode:  {refine_params.protect_mode}")
    print(f"  Separate SH:   {SPARSE_ADAM_AVAILABLE}")
    print("=" * 60)

    # ========== Stage A: Load baseline ==========
    print("\n=== Stage A: Loading baseline model ===")
    gaussians, scene = load_baseline(dataset, args.iteration, opt)
    print(f"  Loaded {gaussians.get_xyz.shape[0]} Gaussians")

    # ========== Stage B: Analysis & Localization ==========
    if not args.skip_analysis:
        region_manager, analysis_results, defect_view_indices = \
            run_analysis_and_localization(
                gaussians, scene, pipeline, background,
                error_params, localization_params, ablation_params,
                output_dir, args.refine_tag,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                max_loc_views=args.max_loc_views,
            )
    else:
        print("\n=== Stage B: Loading existing regions ===")
        regions_dir = os.path.join(output_dir, "regions", args.refine_tag)
        region_manager = RegionManager.load(regions_dir)
        analysis_results = {}
        defect_view_indices = []
        defect_file = os.path.join(output_dir, "analysis", args.refine_tag, "defect_views.json")
        if os.path.exists(defect_file):
            with open(defect_file) as f:
                defect_view_indices = json.load(f)
        print(f"  Loaded regions from {regions_dir}")

    # Save regions
    regions_dir = os.path.join(output_dir, "regions", args.refine_tag)
    region_manager.save(regions_dir)

    # ========== Stage C: Refinement ==========
    if not args.skip_refine:
        run_local_refinement(
            gaussians, scene, pipeline, background,
            region_manager, analysis_results, defect_view_indices,
            opt, refine_params,
            output_dir, args.refine_tag, tb_writer,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )

        # Save refined model
        refine_name = f"refine_{args.refine_tag}"
        refine_path = os.path.join(
            output_dir, "point_cloud", f"iteration_{refine_name}")
        os.makedirs(refine_path, exist_ok=True)
        gaussians.save_ply(os.path.join(refine_path, "point_cloud.ply"))
        print(f"\n  Refined model saved to {refine_path}")

        # Save config
        config = {
            "baseline_iteration": args.iteration,
            "refine_tag": args.refine_tag,
            "refine_iterations": refine_params.refine_iterations,
            "error_params": vars(error_params),
            "localization_params": vars(localization_params),
            "refine_params": vars(refine_params),
            "ablation_params": vars(ablation_params),
        }
        config_path = os.path.join(
            output_dir, f"refine_config_{args.refine_tag}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # ========== Evaluation ==========
    if not args.skip_eval and not args.skip_refine:
        refine_eval(scene, gaussians, pipeline, background,
                    tb_writer, args.refine_tag,
                    separate_sh=SPARSE_ADAM_AVAILABLE)

    if tb_writer:
        tb_writer.flush()
        tb_writer.close()

    print("\n" + "=" * 60)
    print("Local Refinement Pipeline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
