"""
Validation script for PixelSentinel Pix2Pix.

This module evaluates a trained Pix2Pix model on the validation dataset.

Responsibilities:
    - Load trained checkpoint
    - Initialize models
    - Run validation inference
    - Report validation losses

No preprocessing or dataset transformation logic is implemented here.
"""

from __future__ import annotations

import logging

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.config import CONFIG
from colorization.models.discriminator import PatchGANDiscriminator
from colorization.models.generator import Generator
from colorization.training.checkpoint import load_checkpoint
from colorization.training.trainer import Pix2PixTrainer

LOGGER = logging.getLogger(__name__)

__all__ = [
    "main",
]


def build_optimizer(
    model: torch.nn.Module,
) -> Adam:
    """
    Build optimizer required for checkpoint loading.

    Parameters
    ----------
    model:
        Neural network model.

    Returns
    -------
    Adam
        Optimizer instance.
    """

    return Adam(
        model.parameters(),
        lr=CONFIG.training.learning_rate,
        betas=(
            CONFIG.training.beta1,
            CONFIG.training.beta2,
        ),
    )


def create_validation_loader() -> DataLoader:
    """
    Create validation dataloader.

    Dataset implementation is connected from preprocessing modules.

    Returns
    -------
    DataLoader
        Validation dataloader.
    """

    raise NotImplementedError(
        "Validation dataset loader must be connected."
    )

def main() -> None:
    """
    Execute Pix2Pix validation pipeline.
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
        "Validation device: %s",
        device,
    )

    #
    # ---------------------------------------------------------
    # Validation Data
    # ---------------------------------------------------------
    #

    val_loader = create_validation_loader()

    #
    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    #

    generator = Generator()

    discriminator = PatchGANDiscriminator()

    #
    # ---------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------
    #
    # Optimizers are recreated only because the checkpoint
    # loader restores their states.
    #

    generator_optimizer = build_optimizer(
        generator
    )

    discriminator_optimizer = build_optimizer(
        discriminator
    )

    #
    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    #

    trainer = Pix2PixTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device=device,
        scaler=None,
    )

    #
    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    #

    checkpoint_path = (
        CONFIG.checkpoint.best_dir
        / "generator_checkpoint.pth"
    )

    load_checkpoint(
        path=checkpoint_path,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        scaler=None,
        device=device,
    )

    LOGGER.info(
        "Checkpoint loaded successfully."
    )

    #
    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    #

    metrics = trainer.validate_epoch(
        val_loader
    )

    LOGGER.info(
        "Validation Metrics: %s",
        metrics,
    )


if __name__ == "__main__":
    main()