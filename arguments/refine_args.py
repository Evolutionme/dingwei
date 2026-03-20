#
# Local Refinement Parameters for Gaussian Splatting (v2)
# This module defines all hyperparameters for the local refinement pipeline.
# It is designed to be fully compatible with the original arguments system.
#
# v2 additions:
#   - Adaptive threshold params (from GS-LPM)
#   - Ray triangulation / depth backproject / contribution stat params
#   - Density control: grad_ratio, calibrate_opacity (from GS-LPM)
#   - Boundary constraint, prune hysteresis (from CL-Splats)
#   - AblationParams for strategy toggling
#

from arguments import ParamGroup


class ErrorAnalysisParams(ParamGroup):
    """Parameters for Stage B1: error map construction and defect detection."""
    def __init__(self, parser):
        # --- Error map component weights (ablation: toggle individual terms) ---
        self.w_rgb = 1.0           # L1 photometric error weight
        self.w_ssim = 0.5          # Structural (1-SSIM) error weight
        self.w_edge = 0.3          # Edge gradient error weight
        self.w_depth = 0.0         # Depth error weight (0 = disabled by default)
        self.w_lpips = 0.0         # Perceptual LPIPS error weight (0 = disabled, expensive)

        # --- Defect region extraction ---
        self.error_percentile = 90.0       # Top-k percentile threshold for defect pixels
        self.error_abs_threshold = 0.0     # Absolute error threshold (0 = use percentile only)
        self.min_defect_area = 64          # Minimum connected-component area in pixels
        self.mask_dilate_radius = 5        # Morphological dilation radius for defect masks
        self.error_patch_size = 16         # Patch size for local SSIM/error computation

        # --- Adaptive threshold (from GS-LPM) ---
        self.use_adaptive_threshold = False  # Use patch-based adaptive thresholding
        self.adaptive_patch_size = 16        # Patch size for adaptive threshold computation
        self.adaptive_fill_ratio = 0.5       # Min ratio of defect pixels within a patch

        # --- Multi-view consistency ---
        self.min_view_hits = 3             # Minimum number of views a defect must appear in
        self.cross_view_iou_thresh = 0.1   # IoU threshold for cross-view defect association

        super().__init__(parser, "Error Analysis Parameters")


class LocalizationParams(ParamGroup):
    """Parameters for Stage B2-B4: 2D-3D localization and gradient attribution."""
    def __init__(self, parser):
        # --- Projection-based filtering ---
        self.proj_overlap_thresh = 0.3     # Min overlap ratio between Gaussian projection and mask
        self.visibility_thresh = 0.0       # Min opacity contribution for visibility
        self.depth_tolerance = 0.1         # Relative depth tolerance for depth-consistent filtering

        # --- Ray triangulation (from GS-LPM) ---
        self.ray_pair_angle_min = 15.0     # Min angle (degrees) between paired views
        self.ray_pair_angle_max = 60.0     # Max angle (degrees) between paired views
        self.ray_max_pairs = 5             # Max number of paired views per defect view

        # --- Depth backprojection (from CL-Splats) ---
        self.depth_knn_k = 3               # Number of nearest Gaussians per backprojected point
        self.depth_local_radius = 3.0      # Scale-aware radius multiplier for kNN
        self.depth_evidence_pos = 1.0      # Positive evidence weight per hit
        self.depth_evidence_neg = 0.5      # Negative evidence weight per miss

        # --- Contribution statistics (from FlashSplat idea) ---
        self.contribution_min_overlap = 0.1  # Min overlap area ratio for contribution counting

        # --- Gradient attribution weights ---
        self.attr_w_xyz = 1.0              # Weight for position gradient magnitude
        self.attr_w_feat = 0.5             # Weight for feature gradient magnitude
        self.attr_w_opacity = 0.3          # Weight for opacity gradient magnitude
        self.attr_w_scale = 0.3            # Weight for scale gradient magnitude
        self.attr_w_rotation = 0.2         # Weight for rotation gradient magnitude

        # --- Multi-strategy fusion weights ---
        self.w_strategy_ray = 1.0          # Weight for ray triangulation scores
        self.w_strategy_depth = 1.0        # Weight for depth backprojection scores
        self.w_strategy_contrib = 0.5      # Weight for contribution statistics scores
        self.w_strategy_grad = 1.0         # Weight for gradient attribution scores

        # --- Multi-view voting ---
        self.vote_min_views = 2            # Minimum views a Gaussian must be flagged in
        self.vote_score_percentile = 80.0  # Percentile threshold for Gaussian attribution score

        # --- 3D clustering ---
        self.cluster_eps = 0.05            # DBSCAN epsilon (relative to scene extent)
        self.cluster_min_samples = 5       # DBSCAN minimum samples
        self.context_expand_ratio = 0.1    # Context ring expansion (relative to cluster radius)
        self.remove_isolated = True        # Remove single-view-only isolated Gaussians

        super().__init__(parser, "Localization Parameters")


class RefinementParams(ParamGroup):
    """Parameters for Stage C: local refinement optimization."""
    def __init__(self, parser):
        # --- Refinement iterations ---
        self.refine_iterations = 5000      # Number of local refinement iterations

        # --- Learning rate multipliers (relative to original OptimizationParams) ---
        self.target_lr_multiplier = 1.0    # LR multiplier for target region Gaussians
        self.context_lr_multiplier = 0.1   # LR multiplier for context ring Gaussians
        self.protect_lr_multiplier = 0.0   # LR multiplier for protected region (0 = hard freeze)

        # --- Parameter update control (ablation: toggle which params to update) ---
        self.update_xyz = True             # Allow position updates in target region
        self.update_features = True        # Allow SH feature updates in target region
        self.update_opacity = True         # Allow opacity updates in target region
        self.update_scaling = True         # Allow scale updates in target region
        self.update_rotation = True        # Allow rotation updates in target region

        # --- Context region parameter control ---
        self.ctx_update_xyz = False        # Allow position updates in context region
        self.ctx_update_features = True    # Allow SH feature updates in context region
        self.ctx_update_opacity = True     # Allow opacity updates in context region
        self.ctx_update_scaling = False    # Allow scale updates in context region
        self.ctx_update_rotation = False   # Allow rotation updates in context region

        # --- Local densification/pruning (ablation: toggle on/off) ---
        self.local_densify = False         # Enable local densification in target region
        self.local_prune = False           # Enable local pruning in target region
        self.local_densify_from_iter = 100
        self.local_densify_until_iter = 3000
        self.local_densify_interval = 100
        self.local_densify_grad_threshold = 0.0002
        self.local_opacity_reset_interval = 1000

        # --- GS-LPM style density control ---
        self.grad_ratio = 0.5              # Gradient threshold scaling in target region
        self.calibrate_opacity = False     # Enable opacity calibration in target region
        self.calibrate_top_ratio = 0.5     # Fraction of target Gaussians to reset opacity
        self.calibrate_value = 0.01        # Opacity value to reset to (inverse sigmoid)

        # --- Loss weights for refinement ---
        self.lambda_local_rgb = 1.0        # L1 loss weight on target region
        self.lambda_local_ssim = 0.2       # SSIM loss weight on target region
        self.lambda_local_edge = 0.0       # Edge loss weight on target region
        self.lambda_anchor = 0.1           # Anchor regularization weight (protect region)
        self.lambda_context = 0.05         # Context consistency weight
        self.lambda_boundary = 0.0         # Boundary constraint weight (from CL-Splats)
        self.lambda_depth_refine = 0.0     # Depth regularization during refinement

        # --- Protection mode ---
        self.protect_mode = "hard"         # "hard" = freeze, "soft" = anchor regularization
        self.anchor_param_weight = 10.0    # Anchor loss strength when protect_mode="soft"

        # --- CL-Splats style hysteresis pruning ---
        self.prune_hysteresis = 0          # Number of consecutive out-of-bound iters before prune

        # --- View sampling strategy ---
        self.view_sample_strategy = "defect" # "defect" = sample views with defects, "all" = all views
        self.defect_view_weight = 3.0      # Sampling weight boost for views with defects

        super().__init__(parser, "Refinement Parameters")


class AblationParams(ParamGroup):
    """Parameters for controlling which localization strategies are enabled."""
    def __init__(self, parser):
        self.enable_ray_triangulation = True   # Enable ray triangulation (GS-LPM)
        self.enable_depth_backproject = True    # Enable depth backprojection (CL-Splats)
        self.enable_contribution_stat = True    # Enable projection contribution stats (FlashSplat)
        self.enable_gradient_attr = True        # Enable gradient attribution (v1)

        super().__init__(parser, "Ablation Parameters")
