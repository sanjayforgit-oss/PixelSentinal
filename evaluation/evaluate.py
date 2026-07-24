"""
Evaluation entry point for PixelSentinel.

This module runs complete evaluation of the trained Pix2Pix
colorization model.

Responsibilities:
    - Load trained generator
    - Run inference on test dataset
    - Calculate image quality metrics
    - Aggregate evaluation results
    - Save evaluation report

No training logic is included.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import CONFIG
from colorization.models.generator import Generator
from colorization.inference import load_generator
from evaluation.metrics import evaluate_colorization

LOGGER = logging.getLogger(__name__)

__all__ = [
    "evaluate_model",
    "main",
]


def create_test_loader() -> DataLoader:
    """
    Create test dataloader.

    Dataset implementation is connected externally.

    Returns
    -------
    DataLoader
        Test dataloader.
    """

    raise NotImplementedError(
        "Test dataset loader must be connected."
    )


@torch.no_grad()
def evaluate_model(
    generator: Generator,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate generator on test dataset.

    Parameters
    ----------
    generator:
        Trained Pix2Pix generator.

    dataloader:
        Test dataloader.

    device:
        Evaluation device.

    Returns
    -------
    dict[str, float]
        Average evaluation metrics.
    """

    generator.eval()

    metrics_sum: dict[str, float] = {}

    total_samples = 0

    for inputs, targets in dataloader:

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        predictions = generator(
            inputs
        )

        batch_metrics = evaluate_colorization(
            prediction=predictions,
            target=targets,
        )

        batch_size = inputs.size(0)

        for key, value in batch_metrics.items():

            if key not in metrics_sum:
                metrics_sum[key] = 0.0

            metrics_sum[key] += (
                value
                *
                batch_size
            )

        total_samples += batch_size

    if total_samples == 0:
        return {}

    return {
        key: value / total_samples
        for key, value in metrics_sum.items()
    }


def save_report(
    metrics: dict[str, float],
    path: str | Path,
) -> None:
    """
    Save evaluation report.

    Parameters
    ----------
    metrics:
        Evaluation metrics.

    path:
        Output JSON path.
    """

    report_path = Path(path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    LOGGER.info(
        "Evaluation report saved: %s",
        report_path,
    )


def main() -> None:
    """
    Execute complete evaluation pipeline.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    device = torch.device(
        CONFIG.device.device
    )

    LOGGER.info(
        "Evaluation device: %s",
        device,
    )

    test_loader = create_test_loader()

    generator = load_generator(
        checkpoint_path=(
            CONFIG.checkpoint.best_dir
        
            /
            "generator_checkpoint.pth"
        ),
        device=device,
    )

    metrics = evaluate_model(
        generator=generator,
        dataloader=test_loader,
        device=device,
    )

    LOGGER.info(
        "Evaluation Metrics: %s",
        metrics,
    )

    save_report(
        metrics=metrics,
        path=(
            Path("evaluation")
            /
            "results.json"
        ),
    )


if __name__ == "__main__":
    main()