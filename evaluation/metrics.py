"""
PixelSentinel Evaluation Metrics

Implements scientific image quality metrics for evaluating
generated RGB satellite imagery.

Author: Member 3
Project: PixelSentinel
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

warnings.filterwarnings("ignore")

# ==========================================================
# Optional LPIPS Support
# ==========================================================

LPIPS_AVAILABLE = False

try:
    import torch
    import lpips

    LPIPS_AVAILABLE = True

except ImportError:
    LPIPS_AVAILABLE = False


# ==========================================================
# Image Validation
# ==========================================================

def validate_images(
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    """
    Validate prediction and target images.

    Args:
        prediction: Generated RGB image.
        target: Ground truth RGB image.

    Raises:
        ValueError
    """

    if prediction is None:
        raise ValueError("Prediction image is None.")

    if target is None:
        raise ValueError("Target image is None.")

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch:"
            f" {prediction.shape} vs {target.shape}"
        )

    if prediction.ndim != 3:
        raise ValueError(
            "Prediction must be an RGB image."
        )

    if target.ndim != 3:
        raise ValueError(
            "Ground truth must be an RGB image."
        )


# ==========================================================
# Mean Squared Error
# ==========================================================

def mse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Mean Squared Error.
    """

    validate_images(prediction, target)

    error = prediction.astype(np.float32) - target.astype(np.float32)

    return float(np.mean(error ** 2))


# ==========================================================
# Root Mean Squared Error
# ==========================================================

def rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Root Mean Squared Error.
    """

    return float(np.sqrt(mse(prediction, target)))


# ==========================================================
# Mean Absolute Error
# ==========================================================

def mae(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Mean Absolute Error.
    """

    validate_images(prediction, target)

    error = prediction.astype(np.float32) - target.astype(np.float32)

    return float(np.mean(np.abs(error)))

# ==========================================================
# Peak Signal-to-Noise Ratio (PSNR)
# ==========================================================

def psnr(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Higher values indicate better image quality.
    """

    validate_images(prediction, target)

    return float(
        peak_signal_noise_ratio(
            target,
            prediction,
            data_range=255,
        )
    )


# ==========================================================
# Structural Similarity Index (SSIM)
# ==========================================================

def ssim(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Range:
        -1 to 1

    Higher values indicate better structural similarity.
    """

    validate_images(prediction, target)

    return float(
        structural_similarity(
            target,
            prediction,
            channel_axis=2,
            data_range=255,
        )
    )


# ==========================================================
# Spectral Angle Mapper (SAM)
# ==========================================================
def sam(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Spectral Angle Mapper (SAM).

    Lower values are better.
    A perfect match returns 0 degrees.
    """

    validate_images(prediction, target)

    pred = prediction.astype(np.float64)
    gt = target.astype(np.float64)

    pred = pred.reshape(-1, pred.shape[-1])
    gt = gt.reshape(-1, gt.shape[-1])

    dot = np.sum(pred * gt, axis=1)

    pred_norm = np.linalg.norm(pred, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)

    mask = (pred_norm > 1e-8) & (gt_norm > 1e-8)

    if np.sum(mask) == 0:
        return 0.0

    cosine = dot[mask] / (pred_norm[mask] * gt_norm[mask])

    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.arccos(cosine)

    return float(np.degrees(np.mean(angles)))


def ergas(
    prediction: np.ndarray,
    target: np.ndarray,
    ratio: float = 1.0,
) -> float:
    """
    Compute ERGAS (Erreur Relative Globale Adimensionnelle
    de Synthèse), a common remote sensing metric.

    Lower values indicate better quality.
    """

    validate_images(prediction, target)

    pred = prediction.astype(np.float32)
    gt = target.astype(np.float32)

    num_channels = pred.shape[2]
    ergas_sum = 0.0

    for band in range(num_channels):
        rmse_band = np.sqrt(
            np.mean((pred[:, :, band] - gt[:, :, band]) ** 2)
        )

        mean_band = np.mean(gt[:, :, band])

        if mean_band > 0:
            ergas_sum += (rmse_band / mean_band) ** 2

    ergas_value = (100 / ratio) * np.sqrt(ergas_sum / num_channels)

    return float(ergas_value)

# ==========================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ==========================================================

_lpips_model = None


def get_lpips_model():
    """
    Load LPIPS model only once.
    """

    global _lpips_model

    if not LPIPS_AVAILABLE:
        return None

    if _lpips_model is None:

        _lpips_model = lpips.LPIPS(net="alex")

        _lpips_model.eval()

        if torch.cuda.is_available():
            _lpips_model.cuda()

    return _lpips_model


def lpips_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Optional[float]:
    """
    Compute LPIPS perceptual similarity.

    Lower values indicate better perceptual quality.
    """

    if not LPIPS_AVAILABLE:
        return None

    validate_images(prediction, target)

    model = get_lpips_model()

    pred = prediction.astype(np.float32)
    gt = target.astype(np.float32)

    # Normalize to [-1, 1]
    pred = pred / 127.5 - 1.0
    gt = gt / 127.5 - 1.0

    pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)

    if torch.cuda.is_available():
        pred = pred.cuda()
        gt = gt.cuda()

    with torch.no_grad():
        score = model(pred, gt)

    return float(score.item())


# ==========================================================
# Pearson Correlation Coefficient (PCC)
# ==========================================================

def pcc(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Pearson Correlation Coefficient.

    Higher values are better.
    """

    validate_images(prediction, target)

    pred = prediction.astype(np.float32).flatten()
    gt = target.astype(np.float32).flatten()

    if pred.std() == 0 or gt.std() == 0:
        return 0.0

    return float(np.corrcoef(pred, gt)[0, 1])


# ==========================================================
# Universal Image Quality Index (UIQI)
# ==========================================================

def uiqi(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Universal Image Quality Index.

    Higher values indicate better quality.
    """

    validate_images(prediction, target)

    pred = prediction.astype(np.float64)
    gt = target.astype(np.float64)

    x = pred.flatten()
    y = gt.flatten()

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    covariance = np.mean(
        (x - mean_x) * (y - mean_y)
    )

    numerator = (
        4
        * covariance
        * mean_x
        * mean_y
    )

    denominator = (
        (var_x + var_y)
        * (mean_x ** 2 + mean_y ** 2)
    )

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)

# ==========================================================
# Combined Evaluation
# ==========================================================

def evaluate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    prediction : np.ndarray
        Generated RGB image.
    target : np.ndarray
        Ground-truth RGB image.

    Returns
    -------
    Dict[str, float]
        Dictionary containing all computed metrics.
    """

    validate_images(prediction, target)

    results: Dict[str, float] = {}

    # Error Metrics
    results["MSE"] = mse(prediction, target)
    results["RMSE"] = rmse(prediction, target)
    results["MAE"] = mae(prediction, target)

    # Image Quality Metrics
    results["PSNR"] = psnr(prediction, target)
    results["SSIM"] = ssim(prediction, target)

    # Remote Sensing Metrics
    results["SAM"] = sam(prediction, target)
    results["ERGAS"] = ergas(prediction, target)

    # Statistical Metrics
    results["PCC"] = pcc(prediction, target)
    results["UIQI"] = uiqi(prediction, target)

    # Perceptual Metric (Optional)
    lpips_value = lpips_score(prediction, target)

    if lpips_value is not None:
        results["LPIPS"] = lpips_value

    return results


# ==========================================================
# Public API
# ==========================================================

__all__ = [
    "mse",
    "rmse",
    "mae",
    "psnr",
    "ssim",
    "sam",
    "ergas",
    "pcc",
    "uiqi",
    "lpips_score",
    "evaluate_metrics",
]