"""
Checkpoint management utilities for PixelSentinel Pix2Pix.

This module handles saving and loading complete training states including:

- Generator weights
- Discriminator weights
- Optimizer states
- AMP scaler state
- Epoch information
- Best validation metrics

Checkpointing is separated from the trainer to maintain clean
responsibilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from configs.config import CONFIG

LOGGER = logging.getLogger(__name__)

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]


def save_checkpoint(
    path: str | Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    best_metric: float | None = None,
) -> None:
    """
    Save complete training checkpoint.

    Parameters
    ----------
    path:
        Destination checkpoint path.

    epoch:
        Current training epoch.

    generator:
        Generator model.

    discriminator:
        Discriminator model.

    generator_optimizer:
        Generator optimizer.

    discriminator_optimizer:
        Discriminator optimizer.

    scaler:
        AMP gradient scaler.

    best_metric:
        Best validation metric achieved.
    """

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "generator_state_dict": (
            generator.state_dict()
        ),
        "discriminator_state_dict": (
            discriminator.state_dict()
        ),
        "generator_optimizer_state_dict": (
            generator_optimizer.state_dict()
        ),
        "discriminator_optimizer_state_dict": (
            discriminator_optimizer.state_dict()
        ),
        "best_metric": best_metric,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = (
            scaler.state_dict()
        )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    LOGGER.info(
        "Checkpoint saved: %s",
        checkpoint_path,
    )

def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load complete training checkpoint.

    Parameters
    ----------
    path:
        Checkpoint file path.

    generator:
        Generator model.

    discriminator:
        Discriminator model.

    generator_optimizer:
        Generator optimizer.

    discriminator_optimizer:
        Discriminator optimizer.

    scaler:
        AMP gradient scaler.

    device:
        Device where checkpoint tensors should be loaded.

    Returns
    -------
    dict[str, Any]
        Metadata from checkpoint containing epoch and metrics.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    map_location = (
        device
        if device is not None
        else CONFIG.device.device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"],
        strict=False
    )

    discriminator.load_state_dict(
        checkpoint["discriminator_state_dict"]
    )

    try:
        generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state_dict"]
        )
    except ValueError as e:
        LOGGER.warning(
            "Generator architecture changed (added edge_enhancer). "
            "Re-initializing generator optimizer state for new parameters."
        )

    try:
        discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )
    except ValueError as e:
        LOGGER.warning(
            "Discriminator architecture changed (added edge_enhancer). "
            "Re-initializing discriminator optimizer state for new parameters."
        )

    if (
        scaler is not None
        and "scaler_state_dict" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    LOGGER.info(
        "Checkpoint loaded: %s | Epoch: %s",
        checkpoint_path,
        checkpoint.get("epoch", "unknown"),
    )

    return {
        "epoch": checkpoint.get(
            "epoch",
            0,
        ),
        "best_metric": checkpoint.get(
            "best_metric",
            None,
        ),
    }