"""
Evaluation metrics for PixelSentinel.

This module provides quantitative metrics for evaluating
infrared-to-RGB colorization quality.

Metrics implemented:

    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Structural Similarity Index (SSIM)

The module contains only evaluation logic.
No training or inference logic is included.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "calculate_mae",
    "calculate_mse",
    "calculate_psnr",
    "calculate_ssim",
    "evaluate_colorization",
]


def calculate_mae(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Calculate Mean Absolute Error.

    Parameters
    ----------
    prediction:
        Generated RGB image tensor.

    target:
        Ground truth RGB image tensor.

    Returns
    -------
    Tensor
        MAE value.
    """

    return torch.mean(
        torch.abs(
            prediction - target
        )
    )


def calculate_mse(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Calculate Mean Squared Error.

    Parameters
    ----------
    prediction:
        Generated RGB image tensor.

    target:
        Ground truth RGB image tensor.

    Returns
    -------
    Tensor
        MSE value.
    """

    return torch.mean(
        (prediction - target) ** 2
    )


def calculate_psnr(
    prediction: Tensor,
    target: Tensor,
    max_value: float = 1.0,
) -> Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    prediction:
        Generated RGB image tensor.

    target:
        Ground truth RGB image tensor.

    max_value:
        Maximum possible pixel value.

    Returns
    -------
    Tensor
        PSNR value in decibels.
    """

    mse = calculate_mse(
        prediction,
        target,
    )

    if mse == 0:
        return torch.tensor(
            float("inf"),
            device=prediction.device,
        )

    psnr = (
        20
        * math.log10(max_value)
        - 10
        * torch.log10(mse)
    )

    return psnr


def calculate_ssim(
    prediction: Tensor,
    target: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """
    Calculate simplified SSIM metric.

    Parameters
    ----------
    prediction:
        Generated RGB image tensor.

    target:
        Ground truth RGB image tensor.

    window_size:
        Gaussian window size.

    sigma:
        Gaussian standard deviation.

    Returns
    -------
    Tensor
        SSIM score.
    """

    channels = prediction.shape[1]

    coordinates = torch.arange(
        window_size,
        device=prediction.device,
    ).float()

    coordinates -= (
        window_size - 1
    ) / 2

    gaussian = torch.exp(
        -(coordinates ** 2)
        /
        (2 * sigma ** 2)
    )

    gaussian /= gaussian.sum()

    window = (
        gaussian.unsqueeze(1)
        @ gaussian.unsqueeze(0)
    )

    window = window.expand(
        channels,
        1,
        window_size,
        window_size,
    )

    mu_prediction = F.conv2d(
        prediction,
        window,
        padding=window_size // 2,
        groups=channels,
    )

    mu_target = F.conv2d(
        target,
        window,
        padding=window_size // 2,
        groups=channels,
    )

    sigma_prediction = F.conv2d(
        prediction * prediction,
        window,
        padding=window_size // 2,
        groups=channels,
    ) - mu_prediction ** 2

    sigma_target = F.conv2d(
        target * target,
        window,
        padding=window_size // 2,
        groups=channels,
    ) - mu_target ** 2

    sigma_cross = F.conv2d(
        prediction * target,
        window,
        padding=window_size // 2,
        groups=channels,
    ) - (
        mu_prediction * mu_target
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (
        (
            2 * mu_prediction * mu_target
            + c1
        )
        *
        (
            2 * sigma_cross
            + c2
        )
    ) / (
        (
            mu_prediction ** 2
            + mu_target ** 2
            + c1
        )
        *
        (
            sigma_prediction
            + sigma_target
            + c2
        )
    )

    return ssim.mean()


@torch.no_grad()
def evaluate_colorization(
    prediction: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """
    Evaluate generated RGB image.

    Parameters
    ----------
    prediction:
        Generated RGB image.

    target:
        Ground truth RGB image.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """

    mae = calculate_mae(
        prediction,
        target,
    )

    mse = calculate_mse(
        prediction,
        target,
    )

    psnr = calculate_psnr(
        prediction,
        target,
    )

    ssim = calculate_ssim(
        prediction,
        target,
    )

    return {
        "MAE": float(
            mae.item()
        ),
        "MSE": float(
            mse.item()
        ),
        "PSNR": float(
            psnr.item()
        ),
        "SSIM": float(
            ssim.item()
        ),
    }