#
# LocalGaussianModel: Extends GaussianModel with region-aware local operations (v2).
# Does NOT modify the original GaussianModel class.
#
# v2 additions:
#   - compute_boundary_loss() from CL-Splats
#   - calibrate_target_opacity() from GS-LPM
#   - grad_ratio support in local_densify_and_prune()
#   - Hysteresis pruning counter from CL-Splats
#   - Fix _exposure init for load_ply path
#   - Broadcastable gradient mask (handles any param tensor shape)
#

import torch
from torch import nn
import numpy as np
from scene.gaussian_model import GaussianModel
from utils.general_utils import get_expon_lr_func, build_rotation, inverse_sigmoid

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class LocalGaussianModel(GaussianModel):
    """Extends GaussianModel with region-aware local refinement capabilities.

    The key additions:
    1. Region-aware optimizer: different LR multipliers per region
    2. Parameter freezing: granular control over which params update in which region
    3. Local densification/pruning: only in target region (with grad_ratio)
    4. Anchor regularization: keep protected region close to baseline
    5. Boundary constraint loss (from CL-Splats)
    6. Opacity calibration (from GS-LPM)
    """

    def __init__(self, sh_degree, optimizer_type="default"):
        super().__init__(sh_degree, optimizer_type)
        # Region masks
        self._target_mask = None
        self._context_mask = None
        self._protect_mask = None
        # Baseline snapshot for anchor regularization
        self._anchor_xyz = None
        self._anchor_features_dc = None
        self._anchor_features_rest = None
        self._anchor_opacity = None
        self._anchor_scaling = None
        self._anchor_rotation = None
        # Hysteresis pruning counter (from CL-Splats)
        self._out_of_bound_count = None
        # Zone bounding boxes for boundary constraint
        self._zone_bboxes = None

    def set_regions(self, target_mask, context_mask, protect_mask):
        """Set the three-region partition masks."""
        self._target_mask = target_mask
        self._context_mask = context_mask
        self._protect_mask = protect_mask
        # Init hysteresis counter
        self._out_of_bound_count = torch.zeros(
            target_mask.shape[0], dtype=torch.int32, device=target_mask.device)

    def set_zone_bboxes(self, zones):
        """Store 3D zone bounding boxes for boundary constraint."""
        self._zone_bboxes = zones

    def snapshot_baseline(self):
        """Save a frozen copy of current parameters as anchor reference."""
        self._anchor_xyz = self._xyz.detach().clone()
        self._anchor_features_dc = self._features_dc.detach().clone()
        self._anchor_features_rest = self._features_rest.detach().clone()
        self._anchor_opacity = self._opacity.detach().clone()
        self._anchor_scaling = self._scaling.detach().clone()
        self._anchor_rotation = self._rotation.detach().clone()

    def _ensure_exposure(self):
        """Ensure _exposure is initialized (fixes load_ply path where it's missing)."""
        if not hasattr(self, '_exposure') or self._exposure is None:
            num_cams = 1
            if hasattr(self, 'pretrained_exposures') and self.pretrained_exposures is not None:
                num_cams = max(1, len(self.pretrained_exposures))
            self._exposure = nn.Parameter(
                torch.eye(3, 4, device="cuda").unsqueeze(0).repeat(num_cams, 1, 1),
                requires_grad=False)

    def refine_training_setup(self, training_args, refine_args):
        """Setup optimizer for local refinement with region-aware LR.

        Args:
            training_args: Original OptimizationParams
            refine_args: RefinementParams with LR multipliers and update flags
        """
        self._ensure_exposure()

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        target_mult = refine_args.target_lr_multiplier

        # Base learning rates from original training args
        base_lrs = {
            "xyz": training_args.position_lr_init * self.spatial_lr_scale,
            "f_dc": training_args.feature_lr,
            "f_rest": training_args.feature_lr / 20.0,
            "opacity": training_args.opacity_lr,
            "scaling": training_args.scaling_lr,
            "rotation": training_args.rotation_lr,
        }

        # Build parameter groups with region-aware LR
        param_list = [
            {'params': [self._xyz], 'lr': base_lrs["xyz"] * target_mult, "name": "xyz"},
            {'params': [self._features_dc], 'lr': base_lrs["f_dc"] * target_mult, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': base_lrs["f_rest"] * target_mult, "name": "f_rest"},
            {'params': [self._opacity], 'lr': base_lrs["opacity"] * target_mult, "name": "opacity"},
            {'params': [self._scaling], 'lr': base_lrs["scaling"] * target_mult, "name": "scaling"},
            {'params': [self._rotation], 'lr': base_lrs["rotation"] * target_mult, "name": "rotation"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(param_list, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE:
            try:
                self.optimizer = SparseGaussianAdam(param_list, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(param_list, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=base_lrs["xyz"],
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=refine_args.refine_iterations)

        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init, training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=refine_args.refine_iterations)

        # Store refine args for gradient masking
        self._refine_args = refine_args

    def apply_gradient_mask(self):
        """Zero out or scale gradients per region (broadcastable for any param shape).

        Called after loss.backward() and before optimizer.step().
        Uses CL-Splats style: build a [N] multiplier then broadcast to param shape.
        """
        if self._target_mask is None:
            return

        refine = self._refine_args
        protect_mult = refine.protect_lr_multiplier
        context_mult = refine.context_lr_multiplier
        target_mult = max(refine.target_lr_multiplier, 1e-8)

        param_update_config = {
            "_xyz": (refine.update_xyz, refine.ctx_update_xyz),
            "_features_dc": (refine.update_features, refine.ctx_update_features),
            "_features_rest": (refine.update_features, refine.ctx_update_features),
            "_opacity": (refine.update_opacity, refine.ctx_update_opacity),
            "_scaling": (refine.update_scaling, refine.ctx_update_scaling),
            "_rotation": (refine.update_rotation, refine.ctx_update_rotation),
        }

        for param_name, (target_update, ctx_update) in param_update_config.items():
            param = getattr(self, param_name)
            if param.grad is None:
                continue

            # Build per-Gaussian scale factor [N]
            N = param.shape[0]
            scale = torch.ones(N, device=param.device)

            # Protected region
            if protect_mult == 0.0:
                scale[self._protect_mask[:N]] = 0.0
            else:
                scale[self._protect_mask[:N]] = protect_mult / target_mult

            # Context region
            if not ctx_update:
                scale[self._context_mask[:N]] = 0.0
            elif context_mult < target_mult:
                scale[self._context_mask[:N]] = context_mult / target_mult

            # Target region: zero if this param type is disabled
            if not target_update:
                scale[self._target_mask[:N]] = 0.0

            # Broadcast [N] -> param.grad shape (e.g. [N,3], [N,1,C], etc.)
            view_shape = [N] + [1] * (param.grad.dim() - 1)
            param.grad.mul_(scale.view(*view_shape))

    def compute_anchor_loss(self, weight: float = 0.1) -> torch.Tensor:
        """Anchor regularization for protected region (penalize deviation from baseline)."""
        if self._anchor_xyz is None or self._protect_mask is None:
            return torch.tensor(0.0, device="cuda")

        mask = self._protect_mask
        count = mask.sum().float().clamp(min=1.0)

        if count < 1:
            return torch.tensor(0.0, device="cuda")

        loss = (self._xyz[mask] - self._anchor_xyz[mask]).pow(2).sum() / count
        loss += (self._features_dc[mask] - self._anchor_features_dc[mask]).pow(2).sum() / count
        loss += (self._opacity[mask] - self._anchor_opacity[mask]).pow(2).sum() / count
        loss += (self._scaling[mask] - self._anchor_scaling[mask]).pow(2).sum() / count

        return weight * loss

    def compute_context_consistency_loss(self, weight: float = 0.05) -> torch.Tensor:
        """Context consistency loss to avoid boundary artifacts."""
        if self._anchor_xyz is None or self._context_mask is None:
            return torch.tensor(0.0, device="cuda")

        mask = self._context_mask
        count = mask.sum().float().clamp(min=1.0)

        if count < 1:
            return torch.tensor(0.0, device="cuda")

        loss = (self._xyz[mask] - self._anchor_xyz[mask]).pow(2).sum() / count
        loss += 0.5 * (self._features_dc[mask] - self._anchor_features_dc[mask]).pow(2).sum() / count

        return weight * loss

    def compute_boundary_loss(self, weight: float = 0.0) -> torch.Tensor:
        """Boundary constraint: penalize target Gaussians that drift outside zone bboxes.
        (From CL-Splats constraints/primitives.py)

        Uses the zone bounding spheres set by set_zone_bboxes().
        """
        if weight <= 0.0 or self._zone_bboxes is None or not self._zone_bboxes:
            return torch.tensor(0.0, device="cuda")

        if self._target_mask is None or self._target_mask.sum() == 0:
            return torch.tensor(0.0, device="cuda")

        target_xyz = self._xyz[self._target_mask]  # [T, 3]
        if target_xyz.shape[0] == 0:
            return torch.tensor(0.0, device="cuda")

        # Compute min distance to any zone boundary
        min_excess = torch.zeros(target_xyz.shape[0], device="cuda")
        for zone in self._zone_bboxes:
            center = zone["center"].to("cuda")
            radius = zone["radius"]
            dists = (target_xyz - center.unsqueeze(0)).norm(dim=1)
            excess = (dists - radius).clamp(min=0)
            # For multi-zone: each Gaussian only needs to be in ONE zone
            if min_excess.sum() == 0:
                min_excess = excess
            else:
                min_excess = torch.min(min_excess, excess)

        loss = min_excess.pow(2).mean()
        return weight * loss

    def calibrate_target_opacity(self, top_ratio: float = 0.5, value: float = 0.01):
        """Reset opacity of high-opacity target Gaussians to encourage redistribution.
        (From GS-LPM points_calibration)

        Args:
            top_ratio: Fraction of target Gaussians to reset (sorted by opacity descending)
            value: New opacity value (pre-sigmoid)
        """
        if self._target_mask is None or self._target_mask.sum() == 0:
            return

        target_indices = self._target_mask.nonzero(as_tuple=True)[0]
        target_opacities = self.get_opacity[target_indices].squeeze()

        # Select top-ratio by opacity
        k = max(1, int(top_ratio * len(target_indices)))
        _, top_idx = target_opacities.topk(k)
        reset_indices = target_indices[top_idx]

        # Reset opacity to low value
        with torch.no_grad():
            self._opacity[reset_indices] = inverse_sigmoid(
                torch.ones_like(self._opacity[reset_indices]) * value)

    def local_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                                 radii, grad_ratio: float = 1.0,
                                 prune_hysteresis: int = 0):
        """Densification and pruning restricted to target region only.

        Args:
            max_grad: Base gradient threshold
            grad_ratio: Scale factor for gradient threshold in target region (from GS-LPM)
            prune_hysteresis: Number of consecutive iters a Gaussian must be out-of-bound
                              before being pruned (from CL-Splats)
        """
        if self._target_mask is None:
            return

        # Apply grad_ratio: lower threshold in target region for more aggressive densification
        effective_max_grad = max_grad * grad_ratio

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Mask grads to only target region
        target_grads = grads.clone()
        target_grads[~self._target_mask] = 0.0

        self.tmp_radii = radii

        # Clone (only from target, small Gaussians)
        selected_clone = torch.norm(target_grads, dim=-1) >= effective_max_grad
        selected_clone = selected_clone & \
            (torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent)
        selected_clone = selected_clone & self._target_mask

        if selected_clone.sum() > 0:
            new_xyz = self._xyz[selected_clone]
            new_features_dc = self._features_dc[selected_clone]
            new_features_rest = self._features_rest[selected_clone]
            new_opacities = self._opacity[selected_clone]
            new_scaling = self._scaling[selected_clone]
            new_rotation = self._rotation[selected_clone]
            new_tmp_radii = self.tmp_radii[selected_clone]

            self.densification_postfix(
                new_xyz, new_features_dc, new_features_rest,
                new_opacities, new_scaling, new_rotation, new_tmp_radii)

            n_new = selected_clone.sum().item()
            self._extend_region_masks(n_new, is_target=True)
            self._extend_anchor(selected_clone, repeat_n=1)

        # Split (only from target, large Gaussians)
        n_init = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init,), device="cuda")
        padded_grad[:target_grads.shape[0]] = target_grads.squeeze()
        selected_split = padded_grad >= effective_max_grad
        selected_split = selected_split & \
            (torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent)
        selected_split = selected_split & self._target_mask[:n_init]

        if selected_split.sum() > 0:
            N = 2
            stds = self.get_scaling[selected_split].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_split]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                      self.get_xyz[selected_split].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_split].repeat(N, 1) / (0.8 * N))
            new_rotation = self._rotation[selected_split].repeat(N, 1)
            new_features_dc = self._features_dc[selected_split].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_split].repeat(N, 1, 1)
            new_opacity = self._opacity[selected_split].repeat(N, 1)
            new_tmp_radii = self.tmp_radii[selected_split].repeat(N)

            self.densification_postfix(
                new_xyz, new_features_dc, new_features_rest,
                new_opacity, new_scaling, new_rotation, new_tmp_radii)

            n_new = N * selected_split.sum().item()
            self._extend_region_masks(n_new, is_target=True)
            self._extend_anchor(selected_split, repeat_n=N)

            # Prune the original split points
            prune_filter = torch.cat([
                selected_split,
                torch.zeros(n_new, device="cuda", dtype=bool)])
            self._prune_with_regions(prune_filter)

        # Opacity-based pruning in target region only
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = prune_mask & self._target_mask[:prune_mask.shape[0]]

        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            big = (big_vs | big_ws) & self._target_mask[:big_vs.shape[0]]
            prune_mask = prune_mask | big

        # Hysteresis pruning (from CL-Splats): only prune after N consecutive violations
        if prune_hysteresis > 0 and self._out_of_bound_count is not None:
            n = prune_mask.shape[0]
            self._out_of_bound_count[:n] = torch.where(
                prune_mask, self._out_of_bound_count[:n] + 1,
                torch.zeros_like(self._out_of_bound_count[:n]))
            prune_mask = prune_mask & (self._out_of_bound_count[:n] >= prune_hysteresis)

        if prune_mask.sum() > 0:
            self._prune_with_regions(prune_mask)

        self.tmp_radii = None
        torch.cuda.empty_cache()

    def _extend_region_masks(self, n_new: int, is_target: bool = True):
        """Extend region masks for newly added Gaussians."""
        device = self._target_mask.device
        if is_target:
            self._target_mask = torch.cat([
                self._target_mask, torch.ones(n_new, dtype=torch.bool, device=device)])
            self._context_mask = torch.cat([
                self._context_mask, torch.zeros(n_new, dtype=torch.bool, device=device)])
            self._protect_mask = torch.cat([
                self._protect_mask, torch.zeros(n_new, dtype=torch.bool, device=device)])
        else:
            self._target_mask = torch.cat([
                self._target_mask, torch.zeros(n_new, dtype=torch.bool, device=device)])
            self._context_mask = torch.cat([
                self._context_mask, torch.zeros(n_new, dtype=torch.bool, device=device)])
            self._protect_mask = torch.cat([
                self._protect_mask, torch.ones(n_new, dtype=torch.bool, device=device)])

        if self._out_of_bound_count is not None:
            self._out_of_bound_count = torch.cat([
                self._out_of_bound_count,
                torch.zeros(n_new, dtype=torch.int32, device=device)])

    def _extend_anchor(self, selected_mask, repeat_n: int = 1):
        """Extend anchor snapshots for newly cloned/split Gaussians."""
        if self._anchor_xyz is None:
            return
        if repeat_n == 1:
            self._anchor_xyz = torch.cat([self._anchor_xyz, self._anchor_xyz[selected_mask]])
            self._anchor_features_dc = torch.cat([self._anchor_features_dc, self._anchor_features_dc[selected_mask]])
            self._anchor_features_rest = torch.cat([self._anchor_features_rest, self._anchor_features_rest[selected_mask]])
            self._anchor_opacity = torch.cat([self._anchor_opacity, self._anchor_opacity[selected_mask]])
            self._anchor_scaling = torch.cat([self._anchor_scaling, self._anchor_scaling[selected_mask]])
            self._anchor_rotation = torch.cat([self._anchor_rotation, self._anchor_rotation[selected_mask]])
        else:
            self._anchor_xyz = torch.cat([self._anchor_xyz, self._anchor_xyz[selected_mask].repeat(repeat_n, 1)])
            self._anchor_features_dc = torch.cat([self._anchor_features_dc, self._anchor_features_dc[selected_mask].repeat(repeat_n, 1, 1)])
            self._anchor_features_rest = torch.cat([self._anchor_features_rest, self._anchor_features_rest[selected_mask].repeat(repeat_n, 1, 1)])
            self._anchor_opacity = torch.cat([self._anchor_opacity, self._anchor_opacity[selected_mask].repeat(repeat_n, 1)])
            self._anchor_scaling = torch.cat([self._anchor_scaling, self._anchor_scaling[selected_mask].repeat(repeat_n, 1)])
            self._anchor_rotation = torch.cat([self._anchor_rotation, self._anchor_rotation[selected_mask].repeat(repeat_n, 1)])

    def _prune_with_regions(self, mask):
        """Prune points and update region masks consistently."""
        valid = ~mask
        optimizable_tensors = self._prune_optimizer(valid)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid]

        # Update region masks
        self._target_mask = self._target_mask[valid]
        self._context_mask = self._context_mask[valid]
        self._protect_mask = self._protect_mask[valid]

        if self._out_of_bound_count is not None:
            self._out_of_bound_count = self._out_of_bound_count[valid]

        # Update anchor snapshots
        if self._anchor_xyz is not None:
            self._anchor_xyz = self._anchor_xyz[valid]
            self._anchor_features_dc = self._anchor_features_dc[valid]
            self._anchor_features_rest = self._anchor_features_rest[valid]
            self._anchor_opacity = self._anchor_opacity[valid]
            self._anchor_scaling = self._anchor_scaling[valid]
            self._anchor_rotation = self._anchor_rotation[valid]
