#
# Region Utilities for Local Refinement (v2)
# Manages the three-region classification (target/context/protected),
# provides mask I/O and visualization helpers.
#
# v2 changes:
#   - Fix: get_update_mask uses getattr instead of .get()
#   - Add: save defect regions JSON
#   - Add: zone_bboxes saving
#

import torch
import numpy as np
import os
import json
from typing import Dict, Tuple, Optional


class RegionManager:
    """Manages the three-region partition of Gaussians for local refinement.

    Regions:
        - target: core low-quality Gaussians, fully optimized
        - context: boundary ring, lightly optimized
        - protect: rest of scene, frozen or anchor-constrained
    """

    def __init__(
        self,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        protect_mask: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            target_mask: [N] bool
            context_mask: [N] bool
            protect_mask: [N] bool
            scores: [N] float, optional attribution scores
        """
        self.target_mask = target_mask
        self.context_mask = context_mask
        self.protect_mask = protect_mask
        self.scores = scores

        self._validate()

    def _validate(self):
        N = self.target_mask.shape[0]
        assert self.context_mask.shape[0] == N
        assert self.protect_mask.shape[0] == N
        # Ensure mutual exclusivity
        overlap = (self.target_mask & self.context_mask) | \
                  (self.target_mask & self.protect_mask) | \
                  (self.context_mask & self.protect_mask)
        assert not overlap.any(), "Region masks must be mutually exclusive"
        # Ensure full coverage
        covered = self.target_mask | self.context_mask | self.protect_mask
        assert covered.all(), "Region masks must cover all Gaussians"

    @property
    def num_total(self) -> int:
        return self.target_mask.shape[0]

    @property
    def num_target(self) -> int:
        return self.target_mask.sum().item()

    @property
    def num_context(self) -> int:
        return self.context_mask.sum().item()

    @property
    def num_protect(self) -> int:
        return self.protect_mask.sum().item()

    def summary(self) -> str:
        return (
            f"RegionManager: {self.num_total} Gaussians total\n"
            f"  Target:  {self.num_target} ({100*self.num_target/self.num_total:.2f}%)\n"
            f"  Context: {self.num_context} ({100*self.num_context/self.num_total:.2f}%)\n"
            f"  Protect: {self.num_protect} ({100*self.num_protect/self.num_total:.2f}%)"
        )

    def get_lr_multiplier(
        self,
        target_mult: float = 1.0,
        context_mult: float = 0.1,
        protect_mult: float = 0.0,
    ) -> torch.Tensor:
        """Return per-Gaussian learning rate multiplier [N]."""
        mult = torch.zeros(self.num_total, device=self.target_mask.device)
        mult[self.target_mask] = target_mult
        mult[self.context_mask] = context_mult
        mult[self.protect_mask] = protect_mult
        return mult

    def get_update_mask(
        self,
        param_name: str,
        target_params,
        context_params,
    ) -> torch.Tensor:
        """Return per-Gaussian bool mask for whether a given parameter is updatable.

        Args:
            param_name: One of "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"
            target_params: GroupParams or dict with keys like "update_xyz", "update_features", etc.
            context_params: GroupParams or dict with keys like "ctx_update_xyz", etc.

        Returns:
            [N] bool tensor
        """
        param_map = {
            "xyz": ("update_xyz", "ctx_update_xyz"),
            "f_dc": ("update_features", "ctx_update_features"),
            "f_rest": ("update_features", "ctx_update_features"),
            "opacity": ("update_opacity", "ctx_update_opacity"),
            "scaling": ("update_scaling", "ctx_update_scaling"),
            "rotation": ("update_rotation", "ctx_update_rotation"),
        }

        if param_name not in param_map:
            return torch.zeros(self.num_total, dtype=torch.bool, device=self.target_mask.device)

        target_key, ctx_key = param_map[param_name]
        mask = torch.zeros(self.num_total, dtype=torch.bool, device=self.target_mask.device)

        # Support both dict and GroupParams objects
        if isinstance(target_params, dict):
            t_val = target_params.get(target_key, True)
        else:
            t_val = getattr(target_params, target_key, True)

        if isinstance(context_params, dict):
            c_val = context_params.get(ctx_key, False)
        else:
            c_val = getattr(context_params, ctx_key, False)

        if t_val:
            mask |= self.target_mask
        if c_val:
            mask |= self.context_mask

        # Protected region never updated (unless protect_mode=soft, handled elsewhere)
        return mask

    def save(self, path: str):
        """Save region masks and scores to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.target_mask, os.path.join(path, "target_mask.pt"))
        torch.save(self.context_mask, os.path.join(path, "context_mask.pt"))
        torch.save(self.protect_mask, os.path.join(path, "protect_mask.pt"))
        if self.scores is not None:
            torch.save(self.scores, os.path.join(path, "scores.pt"))

        meta = {
            "num_total": self.num_total,
            "num_target": self.num_target,
            "num_context": self.num_context,
            "num_protect": self.num_protect,
        }
        with open(os.path.join(path, "region_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "RegionManager":
        """Load region masks from disk."""
        target_mask = torch.load(os.path.join(path, "target_mask.pt"), map_location=device)
        context_mask = torch.load(os.path.join(path, "context_mask.pt"), map_location=device)
        protect_mask = torch.load(os.path.join(path, "protect_mask.pt"), map_location=device)
        scores_path = os.path.join(path, "scores.pt")
        scores = torch.load(scores_path, map_location=device) if os.path.exists(scores_path) else None
        return cls(target_mask, context_mask, protect_mask, scores)


def save_error_analysis_results(
    analysis_results: Dict[int, dict],
    output_dir: str,
):
    """Save per-view error maps, defect masks, and defect region bboxes."""
    import torchvision
    os.makedirs(output_dir, exist_ok=True)

    error_dir = os.path.join(output_dir, "error_maps")
    mask_dir = os.path.join(output_dir, "defect_masks")
    region_dir = os.path.join(output_dir, "defect_regions")
    os.makedirs(error_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(region_dir, exist_ok=True)

    for vidx, result in analysis_results.items():
        error_map = result["error_map"]
        defect_mask = result["defect_mask"]

        # Normalize error map to [0, 1] for visualization
        e_min, e_max = error_map.min(), error_map.max()
        if e_max > e_min:
            error_vis = (error_map - e_min) / (e_max - e_min)
        else:
            error_vis = error_map
        torchvision.utils.save_image(
            error_vis.unsqueeze(0),
            os.path.join(error_dir, f"{vidx:05d}.png")
        )

        # Save defect mask
        torchvision.utils.save_image(
            defect_mask.float().unsqueeze(0),
            os.path.join(mask_dir, f"{vidx:05d}.png")
        )

        # Save defect regions (bounding boxes) as JSON
        defect_regions = result.get("defect_regions", [])
        if defect_regions:
            with open(os.path.join(region_dir, f"{vidx:05d}.json"), "w") as f:
                json.dump(defect_regions, f, indent=2)


def visualize_regions_on_render(
    render: torch.Tensor,
    gaussians,
    viewpoint_cam,
    region_manager: RegionManager,
    render_pkg: dict,
) -> torch.Tensor:
    """Overlay region colors on a rendered image for visualization.

    Target = red overlay, Context = yellow overlay, Protect = no overlay.

    Args:
        render: [C, H, W] rendered image
        gaussians: GaussianModel
        viewpoint_cam: Camera
        region_manager: RegionManager
        render_pkg: Render output (for visibility)

    Returns:
        vis_image: [C, H, W] visualization image
    """
    from utils.localization import project_gaussians_to_2d

    H, W = render.shape[1], render.shape[2]
    vis = render.clone()

    xy_pixel, _ = project_gaussians_to_2d(gaussians, viewpoint_cam)
    radii = render_pkg["radii"]
    visible = radii > 0

    # Draw dots for target (red) and context (yellow) Gaussians
    for mask, color in [
        (region_manager.target_mask & visible, [1.0, 0.0, 0.0]),
        (region_manager.context_mask & visible, [1.0, 1.0, 0.0]),
    ]:
        indices = mask.nonzero(as_tuple=True)[0]
        for i in indices:
            px = int(xy_pixel[i, 0].item())
            py = int(xy_pixel[i, 1].item())
            r = max(1, min(3, int(radii[i].item() * 0.1)))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        for c in range(3):
                            vis[c, ny, nx] = color[c] * 0.7 + vis[c, ny, nx] * 0.3

    return vis
