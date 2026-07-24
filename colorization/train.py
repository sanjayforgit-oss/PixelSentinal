"""
Training entry point for PixelSentinel Pix2Pix.

This module launches the complete Pix2Pix training pipeline.

Responsibilities:
    - Load configuration
    - Prepare datasets
    - Initialize models
    - Initialize optimizers
    - Initialize AMP
    - Run training epochs
    - Save checkpoints

Dataset implementation is assumed to exist in preprocessing modules.
No preprocessing logic is implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.config import CONFIG
from colorization.models.discriminator import PatchGANDiscriminator
from colorization.models.generator import Generator
from colorization.models.weights import initialize_weights
from colorization.training.amp import create_grad_scaler
from colorization.training.checkpoint import save_checkpoint
from colorization.training.trainer import Pix2PixTrainer

LOGGER = logging.getLogger(__name__)

__all__ = [
    "main",
]


def build_optimizer(
    model: torch.nn.Module,
) -> Adam:
    """
    Create Adam optimizer following Pix2Pix settings.

    Parameters
    ----------
    model:
        Neural network model.

    Returns
    -------
    Adam
        Configured optimizer.
    """

    return Adam(
        model.parameters(),
        lr=CONFIG.training.learning_rate,
        betas=(
            CONFIG.training.beta1,
            CONFIG.training.beta2,
        ),
    )


def create_dataloaders() -> tuple[
    DataLoader,
    DataLoader,
]:
    """
    Create training and validation dataloaders.

    Dataset loading is intentionally separated from this module.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation loaders.
    """

    raise NotImplementedError(
        "Dataset loader must be connected from preprocessing module."
    )

def main() -> None:
    """
    Main training pipeline.

    Initializes models, optimizers, AMP, trainer,
    and executes the training loop.
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
        "Using device: %s",
        device,
    )

    #
    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    #

    train_loader, val_loader = create_dataloaders()

    #
    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    #

    generator = Generator()

    discriminator = PatchGANDiscriminator()

    initialize_weights(generator)
    initialize_weights(discriminator)

    #
    # ---------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------
    #

    generator_optimizer = build_optimizer(
        generator
    )

    discriminator_optimizer = build_optimizer(
        discriminator
    )

    #
    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------
    #

    scaler = create_grad_scaler()

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
        scaler=scaler,
    )

    #
    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    #

    best_loss = float("inf")

    epochs = CONFIG.training.epochs

    checkpoint_dir = CONFIG.checkpoint
    

    for epoch in range(epochs):

        LOGGER.info(
            "Starting epoch %d/%d",
            epoch + 1,
            epochs,
        )

        train_metrics = trainer.fit_epoch(
            train_loader
        )

        val_metrics = trainer.validate_epoch(
            val_loader
        )

        LOGGER.info(
            "Train Loss: %s",
            train_metrics,
        )

        LOGGER.info(
            "Validation Loss: %s",
            val_metrics,
        )

        val_loss = val_metrics.get(
            "generator_loss",
            float("inf"),
        )

        if val_loss < best_loss:

            best_loss = val_loss

            save_checkpoint(
                path=(
                    checkpoint_dir.best_dir
                    / "generator_checkpoint.pth"
                ),
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=(
                    generator_optimizer
                ),
                discriminator_optimizer=(
                    discriminator_optimizer
                ),
                scaler=scaler,
                best_metric=best_loss,
            )

            LOGGER.info(
                "Best checkpoint saved."
            )


if __name__ == "__main__":
    main()