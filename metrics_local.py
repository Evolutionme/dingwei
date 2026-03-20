#
# Local Region Metrics for Gaussian Splatting Refinement (v2)
#
# Extends the original metrics.py with region-aware evaluation:
# - Full-image metrics (same as original, for compatibility)
# - Defect-region-only metrics (to measure local improvement)
# - Non-defect-region metrics (to measure global stability)
# - Baseline vs refined comparison summary
#
# Usage:
#   python metrics_local.py -m <model_path> --refine_tag <tag>
#
# The original metrics.py is NOT modified.
#
# v2 changes:
#   - Handle SSIM returning map or scalar
#   - Add baseline vs refined delta summary
#   - Safer mask loading with size validation
#

import os
import json
import torch
import torchvision.transforms.functional as tf
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def load_mask(mask_path: str, H: int, W: int) -> torch.Tensor:
    """Load a defect mask image and convert to bool tensor [H, W]."""
    if not os.path.exists(mask_path):
        return None
    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img.resize((W, H), Image.NEAREST))
    return torch.from_numpy((mask_np > 127).astype(bool))


def compute_masked_metrics(
    render: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compute PSNR, SSIM, LPIPS on a masked region.

    Args:
        render: [1, C, H, W] rendered image
        gt: [1, C, H, W] ground truth image
        mask: [H, W] bool tensor

    Returns:
        Dict with psnr, ssim, lpips values
    """
    if mask is None or mask.sum() == 0:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

    # Expand mask to [1, 1, H, W]
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float().to(render.device)

    # Mask the images (set non-mask pixels to 0)
    render_masked = render * mask_4d
    gt_masked = gt * mask_4d

    # PSNR on masked region
    mse = ((render_masked - gt_masked) ** 2).sum() / (mask.sum() * 3 + 1e-8)
    psnr_val = -10.0 * torch.log10(mse + 1e-8).item()

    # SSIM on masked region
    ssim_result = ssim(render_masked.squeeze(0), gt_masked.squeeze(0))
    if isinstance(ssim_result, torch.Tensor) and ssim_result.dim() >= 2:
        ssim_val = (ssim_result * mask_4d.squeeze(0)).sum() / (mask_4d.sum() + 1e-8)
        ssim_val = ssim_val.item()
    elif isinstance(ssim_result, torch.Tensor):
        ssim_val = ssim_result.mean().item()
    else:
        ssim_val = float(ssim_result)

    # LPIPS on masked region
    lpips_val = lpips(render_masked, gt_masked, net_type='vgg').item()

    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def evaluate_local(model_paths, refine_tag="refine"):
    """Evaluate both baseline and refined models with region-aware metrics.

    For each scene:
    1. Compute full-image metrics (standard)
    2. Compute defect-region metrics (local improvement)
    3. Compute non-defect-region metrics (global stability)
    """
    results = {}

    for scene_dir in model_paths:
        print(f"\nScene: {scene_dir}")
        results[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"
        mask_dir = Path(scene_dir) / "analysis" / refine_tag / "defect_masks"

        if not test_dir.exists():
            print(f"  No test directory found at {test_dir}")
            continue

        for method in sorted(os.listdir(test_dir)):
            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"

            if not renders_dir.exists() or not gt_dir.exists():
                continue

            print(f"  Method: {method}")
            results[scene_dir][method] = {}

            # Full image metrics
            full_psnrs, full_ssims, full_lpipss = [], [], []
            # Defect region metrics
            defect_psnrs, defect_ssims, defect_lpipss = [], [], []
            # Non-defect region metrics
            clean_psnrs, clean_ssims, clean_lpipss = [], [], []

            fnames = sorted(os.listdir(renders_dir))
            for idx, fname in enumerate(tqdm(fnames, desc="  Evaluating")):
                render_img = tf.to_tensor(Image.open(renders_dir / fname)).unsqueeze(0)[:, :3].cuda()
                gt_img = tf.to_tensor(Image.open(gt_dir / fname)).unsqueeze(0)[:, :3].cuda()

                H, W = render_img.shape[2], render_img.shape[3]

                # Full image metrics
                ssim_result = ssim(render_img.squeeze(0), gt_img.squeeze(0))
                if isinstance(ssim_result, torch.Tensor) and ssim_result.dim() >= 2:
                    full_ssims.append(ssim_result.mean().item())
                elif isinstance(ssim_result, torch.Tensor):
                    full_ssims.append(ssim_result.mean().item())
                else:
                    full_ssims.append(float(ssim_result))
                full_psnrs.append(psnr(render_img.squeeze(0), gt_img.squeeze(0)).mean().item())
                full_lpipss.append(lpips(render_img, gt_img, net_type='vgg').item())

                # Load defect mask if available
                mask_path = mask_dir / f"{idx:05d}.png"
                defect_mask = load_mask(str(mask_path), H, W)

                if defect_mask is not None and defect_mask.sum() > 0:
                    # Defect region metrics
                    d_metrics = compute_masked_metrics(render_img, gt_img, defect_mask)
                    defect_psnrs.append(d_metrics["psnr"])
                    defect_ssims.append(d_metrics["ssim"])
                    defect_lpipss.append(d_metrics["lpips"])

                    # Non-defect region metrics
                    clean_mask = ~defect_mask
                    c_metrics = compute_masked_metrics(render_img, gt_img, clean_mask)
                    clean_psnrs.append(c_metrics["psnr"])
                    clean_ssims.append(c_metrics["ssim"])
                    clean_lpipss.append(c_metrics["lpips"])

            # Aggregate
            full_results = {
                "PSNR": float(np.mean(full_psnrs)) if full_psnrs else 0.0,
                "SSIM": float(np.mean(full_ssims)) if full_ssims else 0.0,
                "LPIPS": float(np.mean(full_lpipss)) if full_lpipss else 0.0,
            }
            defect_results = {
                "PSNR": float(np.mean(defect_psnrs)) if defect_psnrs else 0.0,
                "SSIM": float(np.mean(defect_ssims)) if defect_ssims else 0.0,
                "LPIPS": float(np.mean(defect_lpipss)) if defect_lpipss else 0.0,
            }
            clean_results = {
                "PSNR": float(np.mean(clean_psnrs)) if clean_psnrs else 0.0,
                "SSIM": float(np.mean(clean_ssims)) if clean_ssims else 0.0,
                "LPIPS": float(np.mean(clean_lpipss)) if clean_lpipss else 0.0,
            }

            results[scene_dir][method] = {
                "full": full_results,
                "defect_region": defect_results,
                "clean_region": clean_results,
            }

            print(f"    Full  - PSNR: {full_results['PSNR']:.4f}  "
                  f"SSIM: {full_results['SSIM']:.6f}  "
                  f"LPIPS: {full_results['LPIPS']:.6f}")
            if defect_psnrs:
                print(f"    Defect- PSNR: {defect_results['PSNR']:.4f}  "
                      f"SSIM: {defect_results['SSIM']:.6f}  "
                      f"LPIPS: {defect_results['LPIPS']:.6f}")
                print(f"    Clean - PSNR: {clean_results['PSNR']:.4f}  "
                      f"SSIM: {clean_results['SSIM']:.6f}  "
                      f"LPIPS: {clean_results['LPIPS']:.6f}")

        # Save results
        output_path = os.path.join(scene_dir, f"results_local_{refine_tag}.json")
        with open(output_path, "w") as f:
            json.dump(results[scene_dir], f, indent=2)
        print(f"  Results saved to {output_path}")

        # Print baseline vs refined delta if both exist
        methods = list(results[scene_dir].keys())
        if len(methods) >= 2:
            m0, m1 = methods[0], methods[-1]
            r0 = results[scene_dir][m0]
            r1 = results[scene_dir][m1]
            if "full" in r0 and "full" in r1:
                dp = r1["full"]["PSNR"] - r0["full"]["PSNR"]
                ds = r1["full"]["SSIM"] - r0["full"]["SSIM"]
                dl = r1["full"]["LPIPS"] - r0["full"]["LPIPS"]
                print(f"  Delta ({m1} - {m0}):")
                print(f"    Full  dPSNR={dp:+.4f}  dSSIM={ds:+.6f}  dLPIPS={dl:+.6f}")
            if "defect_region" in r0 and "defect_region" in r1:
                dp = r1["defect_region"]["PSNR"] - r0["defect_region"]["PSNR"]
                ds = r1["defect_region"]["SSIM"] - r0["defect_region"]["SSIM"]
                dl = r1["defect_region"]["LPIPS"] - r0["defect_region"]["LPIPS"]
                print(f"    Defect dPSNR={dp:+.4f}  dSSIM={ds:+.6f}  dLPIPS={dl:+.6f}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Local Region Metrics")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str)
    parser.add_argument('--refine_tag', type=str, default="refine",
                        help="Tag matching the refinement run to evaluate")
    args = parser.parse_args()

    evaluate_local(args.model_paths, args.refine_tag)
