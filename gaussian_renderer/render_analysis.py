#
# Render Analysis Module for Local Refinement (v2)
# Wraps the original render function with additional analysis capabilities:
# - masked rendering loss computation
# - per-view defect-weighted loss
# - region-aware loss aggregation
#
# v2 changes:
#   - Add boundary loss from CL-Splats
#   - Proper masked SSIM via element-wise weighting
#   - Fix GroupParams access via getattr
#   - Exposure handling guard
#

import torch
import math
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


def _getparam(params, name, default):
    """Safely get a parameter from either a GroupParams object or a dict."""
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def render_with_local_loss(
    viewpoint_cam,
    gaussians,
    pipe,
    background: torch.Tensor,
    defect_mask: torch.Tensor = None,
    region_manager=None,
    refine_params=None,
    separate_sh: bool = False,
    use_trained_exp: bool = False,
):
    """Render a view and compute region-aware losses.

    This wraps the original render() without modifying it, and adds:
    - Local loss: only on defect mask region
    - Global loss: on full image (weighted down for anchor)
    - Anchor loss: parameter-space regularization for protected region
    - Context consistency loss: for context ring
    - Boundary loss: CL-Splats zone constraint

    Args:
        viewpoint_cam: Camera object
        gaussians: LocalGaussianModel or GaussianModel
        pipe: PipelineParams
        background: Background tensor
        defect_mask: [H, W] bool tensor for this view (None = full image loss)
        region_manager: RegionManager instance (optional)
        refine_params: RefinementParams GroupParams or dict (optional)
        separate_sh: SH mode
        use_trained_exp: Exposure mode

    Returns:
        Dict with:
            "render_pkg": original render output
            "image": rendered image [C, H, W]
            "gt": ground truth [C, H, W]
            "loss_local": local masked loss (scalar)
            "loss_anchor": anchor regularization loss (scalar)
            "loss_context": context consistency loss (scalar)
            "loss_boundary": boundary constraint loss (scalar)
            "loss_total": combined loss (scalar)
    """
    # Standard render
    render_pkg = render(
        viewpoint_cam, gaussians, pipe, background,
        use_trained_exp=use_trained_exp,
        separate_sh=separate_sh
    )
    image = render_pkg["render"]
    gt = viewpoint_cam.original_image[:3, :, :].cuda()

    # Apply alpha mask if present
    if viewpoint_cam.alpha_mask is not None:
        alpha_mask = viewpoint_cam.alpha_mask.cuda()
        image = image * alpha_mask

    # Extract loss weights (supports both GroupParams and dict)
    lambda_rgb = _getparam(refine_params, "lambda_local_rgb", 1.0)
    lambda_ssim = _getparam(refine_params, "lambda_local_ssim", 0.2)
    lambda_anchor = _getparam(refine_params, "lambda_anchor", 0.1)
    lambda_context = _getparam(refine_params, "lambda_context", 0.05)
    lambda_boundary = _getparam(refine_params, "lambda_boundary", 0.0)

    # --- Local masked loss ---
    if defect_mask is not None and defect_mask.any():
        mask_float = defect_mask.float().unsqueeze(0)  # [1, H, W]
        mask_pixels = mask_float.sum() * 3  # total scalar values in mask

        # L1 on masked region
        masked_l1 = (torch.abs(image - gt) * mask_float).sum() / (mask_pixels + 1e-8)

        # Masked SSIM: compute full SSIM map, then weight by mask
        ssim_map = ssim(image, gt)
        if isinstance(ssim_map, torch.Tensor) and ssim_map.dim() >= 2:
            # ssim returned a map — weight it
            masked_ssim = (ssim_map * mask_float).sum() / (mask_float.sum() + 1e-8)
        else:
            # ssim returned a scalar — use directly
            masked_ssim = ssim_map

        loss_local = lambda_rgb * masked_l1 + lambda_ssim * (1.0 - masked_ssim)
    else:
        # Fallback: full-image loss
        Ll1 = l1_loss(image, gt)
        ssim_val = ssim(image, gt)
        if isinstance(ssim_val, torch.Tensor) and ssim_val.dim() >= 2:
            ssim_val = ssim_val.mean()
        loss_local = lambda_rgb * Ll1 + lambda_ssim * (1.0 - ssim_val)

    # --- Anchor regularization loss ---
    loss_anchor = torch.tensor(0.0, device="cuda")
    if hasattr(gaussians, 'compute_anchor_loss') and lambda_anchor > 0:
        loss_anchor = gaussians.compute_anchor_loss(weight=lambda_anchor)

    # --- Context consistency loss ---
    loss_context = torch.tensor(0.0, device="cuda")
    if hasattr(gaussians, 'compute_context_consistency_loss') and lambda_context > 0:
        loss_context = gaussians.compute_context_consistency_loss(weight=lambda_context)

    # --- Boundary constraint loss (CL-Splats) ---
    loss_boundary = torch.tensor(0.0, device="cuda")
    if hasattr(gaussians, 'compute_boundary_loss') and lambda_boundary > 0:
        loss_boundary = gaussians.compute_boundary_loss(weight=lambda_boundary)

    # --- Total loss ---
    loss_total = loss_local + loss_anchor + loss_context + loss_boundary

    return {
        "render_pkg": render_pkg,
        "image": image,
        "gt": gt,
        "loss_local": loss_local,
        "loss_anchor": loss_anchor,
        "loss_context": loss_context,
        "loss_boundary": loss_boundary,
        "loss_total": loss_total,
    }


def compute_view_defect_weight(
    defect_mask: torch.Tensor,
    base_weight: float = 1.0,
    defect_boost: float = 3.0,
) -> float:
    """Compute sampling weight for a view based on its defect coverage.

    Views with more defects get higher sampling probability during refinement.

    Args:
        defect_mask: [H, W] bool tensor
        base_weight: Base sampling weight for all views
        defect_boost: Extra weight multiplier for defect coverage ratio

    Returns:
        Sampling weight (float)
    """
    if defect_mask is None:
        return base_weight
    ratio = defect_mask.float().mean().item()
    return base_weight + defect_boost * ratio
