"""
Prediction entry point for PixelSentinel Pix2Pix.

This module provides a command-line inference interface.

Responsibilities:
    - Load input tensor
    - Load trained Pix2Pix generator
    - Generate RGB output
    - Save prediction tensor

No preprocessing or visualization logic is implemented here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from configs.config import CONFIG
from colorization.inference import inference_from_checkpoint

LOGGER = logging.getLogger(__name__)

__all__ = [
    "main",
]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "PixelSentinel Pix2Pix "
            "infrared-to-RGB prediction"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to input infrared tensor "
            "(.pt file)"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to trained generator "
            "checkpoint"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to save generated RGB tensor"
        ),
    )

    return parser.parse_args()


def load_input_tensor(
    path: str | Path,
) -> torch.Tensor:
    """
    Load infrared input tensor.

    Parameters
    ----------
    path:
        Tensor file path.

    Returns
    -------
    torch.Tensor
        Input tensor.
    """

    tensor = torch.load(
        path,
        map_location="cpu",
    )

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    return tensor.float()


def save_prediction(
    tensor: torch.Tensor,
    path: str | Path,
) -> None:
    """
    Save generated tensor.

    Parameters
    ----------
    tensor:
        Generated RGB tensor.

    path:
        Output path.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        tensor.cpu(),
        output_path,
    )

    LOGGER.info(
        "Prediction saved: %s",
        output_path,
    )


def main() -> None:
    """
    Execute prediction pipeline.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    args = parse_arguments()

    device = torch.device(
        CONFIG.device.device
    )

    LOGGER.info(
        "Using device: %s",
        device,
    )

    input_tensor = load_input_tensor(
        args.input
    )

    prediction = inference_from_checkpoint(
        checkpoint_path=args.checkpoint,
        input_tensor=input_tensor,
        device=device,
    )

    save_prediction(
        tensor=prediction,
        path=args.output,
    )


if __name__ == "__main__":
    main()