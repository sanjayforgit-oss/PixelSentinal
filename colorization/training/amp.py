"""
Automatic Mixed Precision utilities for PixelSentinel Pix2Pix.

This module provides reusable AMP utilities for GPU accelerated training.

Responsibilities:
    - Create GradScaler
    - Configure autocast settings
    - Maintain clean AMP integration

Training logic remains inside trainer.py.
"""

from __future__ import annotations

import torch

from configs.config import CONFIG

__all__ = [
    "create_grad_scaler",
    "autocast_context",
]


def create_grad_scaler() -> torch.amp.GradScaler | None:
    """
    Create AMP gradient scaler.

    Returns
    -------
    torch.amp.GradScaler | None
        Enabled GradScaler when mixed precision is active,
        otherwise None.
    """

    use_amp = bool(
        getattr(
            CONFIG.training,
            "mixed_precision",
            False,
        )
    )

    if not use_amp:
        return None

    if not torch.cuda.is_available():
        return None

    return torch.amp.GradScaler(
        "cuda"
    )


def autocast_context(
    device: torch.device,
):
    """
    Create automatic mixed precision context.

    Parameters
    ----------
    device:
        Training device.

    Returns
    -------
    Context manager
        AMP autocast context.
    """

    enabled = bool(
        getattr(
            CONFIG.training,
            "mixed_precision",
            False,
        )
    )

    device_type = (
        "cuda"
        if device.type == "cuda"
        else "cpu"
    )

    return torch.amp.autocast(
        device_type=device_type,
        enabled=enabled,
    )