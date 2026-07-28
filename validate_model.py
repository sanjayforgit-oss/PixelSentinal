"""
Validate a PixelSentinel Pix2Pix generator checkpoint.

This script evaluates the generator on the validation split and reports:

- L1 loss
- MSE
- PSNR
- SSIM
- Pixel accuracy
- Pixel F1 score

Important:
Accuracy and F1 are not standard metrics for image-to-image regression.
They are computed here by thresholding normalized pixels into a binary mask.
That makes them usable for a rough sanity check, but PSNR/SSIM/L1 are usually
more meaningful for this task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from colorization.dataset import PixelSentinelDataset
from colorization.models.generator import Generator
from colorization.training.checkpoint import load_checkpoint


def find_split_dir(base_dir: Path, split: str) -> Path:
    candidates = [
        base_dir / split,
        base_dir / split / "input",
        base_dir / split / "target",
        base_dir / split / "inputs",
        base_dir / split / "targets",
    ]
    for candidate in candidates:
        if candidate.exists():
            return base_dir / split
    raise FileNotFoundError(f"Could not find split directory for '{split}' under {base_dir}")


def minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_simple(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Lightweight SSIM approximation over the whole image.
    This is not a full sliding-window SSIM, but it is dependency-free.
    """
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    mu_x = pred_flat.mean(dim=1)
    mu_y = target_flat.mean(dim=1)
    sigma_x = pred_flat.var(dim=1, unbiased=False)
    sigma_y = target_flat.var(dim=1, unbiased=False)
    sigma_xy = ((pred_flat - mu_x.unsqueeze(1)) * (target_flat - mu_y.unsqueeze(1))).mean(dim=1)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return numerator / (denominator + eps)


def binary_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    pred_bin = (pred >= threshold)
    target_bin = (target >= threshold)

    tp = (pred_bin & target_bin).sum().item()
    tn = ((~pred_bin) & (~target_bin)).sum().item()
    fp = (pred_bin & (~target_bin)).sum().item()
    fn = ((~pred_bin) & target_bin).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return accuracy, f1


def evaluate(generator: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    generator.eval()

    total_l1 = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = generator(inputs)

            preds_n = minmax_normalize(preds)
            targets_n = minmax_normalize(targets)

            l1 = F.l1_loss(preds_n, targets_n).item()
            mse = F.mse_loss(preds_n, targets_n).item()
            batch_psnr = psnr(preds_n, targets_n).mean().item()
            batch_ssim = ssim_simple(preds_n, targets_n).mean().item()

            acc, f1 = binary_metrics(preds_n, targets_n)

            total_l1 += l1
            total_mse += mse
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_acc += acc
            total_f1 += f1
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Validation loader produced no batches.")

    return {
        "l1": total_l1 / num_batches,
        "mse": total_mse / num_batches,
        "psnr": total_psnr / num_batches,
        "ssim": total_ssim / num_batches,
        "accuracy": total_acc / num_batches,
        "f1": total_f1 / num_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PixelSentinel generator checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_checkpoint.pth")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    val_dir = find_split_dir(data_root, "val")

    dataset = PixelSentinelDataset(str(val_dir), is_train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    generator = Generator().to(device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "generator_state_dict" in checkpoint:
            generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain 'generator_state_dict'."
            )
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metrics = evaluate(generator, dataloader, device)

    print("\nValidation Results")
    print("-" * 60)
    print(f"L1 Loss     : {metrics['l1']:.6f}")
    print(f"MSE         : {metrics['mse']:.6f}")
    print(f"PSNR (dB)   : {metrics['psnr']:.4f}")
    print(f"SSIM        : {metrics['ssim']:.4f}")
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"F1 Score    : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
