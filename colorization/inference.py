"""
Inference pipeline for PixelSentinel Pix2Pix.

This module performs infrared-to-RGB image translation using a trained
Pix2Pix Generator.

Responsibilities:
    - Load trained generator checkpoint
    - Prepare input tensors
    - Run inference
    - Return generated RGB output

No preprocessing or visualization logic is implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from configs.config import CONFIG
from colorization.models.generator import Generator

LOGGER = logging.getLogger(__name__)

__all__ = [
    "load_generator",
    "run_inference",
]


def load_generator(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Generator:
    """
    Load trained generator model.

    Parameters
    ----------
    checkpoint_path:
        Path to generator checkpoint.

    device:
        Inference device.

    Returns
    -------
    Generator
        Loaded generator model.
    """

    generator = Generator()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "generator_state_dict" in checkpoint:

        generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )

    else:

        generator.load_state_dict(
            checkpoint
        )

    generator.to(device)

    generator.eval()

    LOGGER.info(
        "Generator loaded from %s",
        checkpoint_path,
    )

    return generator


@torch.no_grad()
def run_inference(
    generator: Generator,
    input_tensor: Tensor,
    device: torch.device,
) -> Tensor:
    """
    Generate RGB image from infrared input.

    Parameters
    ----------
    generator:
        Trained Pix2Pix generator.

    input_tensor:
        Infrared tensor.

    device:
        Inference device.

    Returns
    -------
    Tensor
        Generated RGB tensor.
    """

    input_tensor = input_tensor.to(
        device,
        non_blocking=True,
    )

    amp_enabled = bool(
        getattr(
            CONFIG.training,
            "mixed_precision",
            False,
        )
    )

    with torch.amp.autocast(
        device_type=(
            "cuda"
            if device.type == "cuda"
            else "cpu"
        ),
        enabled=amp_enabled,
    ):

        output = generator(
            input_tensor
        )

    return output

def inference_from_checkpoint(
    checkpoint_path: str | Path,
    input_tensor: Tensor,
    device: torch.device | None = None,
) -> Tensor:
    """
    Complete inference pipeline using a checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to trained generator checkpoint.

    input_tensor:
        Infrared input tensor.

    device:
        Inference device.
        If None, uses configured device.

    Returns
    -------
    Tensor
        Generated RGB output tensor.
    """

    if device is None:
        device = torch.device(
            CONFIG.device.device
        )

    generator = load_generator(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    output = run_inference(
        generator=generator,
        input_tensor=input_tensor,
        device=device,
    )

    return output