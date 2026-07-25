"""
Training entry point for PixelSentinel Pix2Pix.

This module launches the complete Pix2Pix training pipeline.

Responsibilities:
    - Load configuration
    - Prepare datasets via colorization.dataset (DDP Ready)
    - Initialize models & weights
    - Initialize optimizers & AMP
    - Support multi-node DDP GPU training (e.g., connected RTX GPUs)
    - Run training & validation loops
    - Save checkpoints
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.config import CONFIG
from colorization.dataset import get_dataloaders
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


def setup_ddp() -> tuple[bool, int, torch.device]:
    """
    Initializes Distributed Data Parallel (DDP) if process group envs are present.
    Fallback to single GPU / CPU if DDP is not active.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
            
        is_distributed = True
        if rank == 0:
            LOGGER.info("DDP process group initialized successfully across %s processes.", os.environ['WORLD_SIZE'])
    else:
        is_distributed = False
        rank = 0
        device = torch.device(CONFIG.device.device if hasattr(CONFIG, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))
        LOGGER.info("Running in single-device mode on device: %s", device)

    return is_distributed, rank, device


def cleanup_ddp(is_distributed: bool) -> None:
    """Closes the process group if DDP is initialized."""
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def build_optimizer(
    model: nn.Module,
) -> Adam:
    """
    Create Adam optimizer following Pix2Pix settings.
    """
    return Adam(
        model.parameters(),
        lr=CONFIG.training.learning_rate,
        betas=(
            CONFIG.training.beta1,
            CONFIG.training.beta2,
        ),
    )


def create_dataloaders(is_distributed: bool = False) -> tuple[
    DataLoader,
    DataLoader,
]:
    """
    Create training and validation dataloaders connecting directly to colorization/dataset.py.
    """
    loaders = get_dataloaders(
        root_dir="datasets",
        batch_size=CONFIG.training.batch_size,
        num_workers=getattr(CONFIG.training, "num_workers", 4),
        is_distributed=is_distributed
    )
    return loaders["train"], loaders["val"]


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

    is_distributed, rank, device = setup_ddp()

    #
    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    #
    train_loader, val_loader = create_dataloaders(is_distributed=is_distributed)

    #
    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    #
    generator = Generator()
    discriminator = PatchGANDiscriminator()

    initialize_weights(generator)
    initialize_weights(discriminator)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Wrap models with DDP if distributed environment is detected
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        generator = DDP(generator, device_ids=[local_rank], output_device=local_rank)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

    #
    # ---------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------
    #
    generator_optimizer = build_optimizer(generator)
    discriminator_optimizer = build_optimizer(discriminator)

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
        # Synchronize multi-node DDP sampler seeds
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            LOGGER.info(
                "Starting epoch %d/%d",
                epoch + 1,
                epochs,
            )

        train_metrics = trainer.fit_epoch(train_loader)
        val_metrics = trainer.validate_epoch(val_loader)

        if rank == 0:
            LOGGER.info("Train Loss: %s", train_metrics)
            LOGGER.info("Validation Loss: %s", val_metrics)

        val_loss = val_metrics.get("generator_loss", float("inf"))

        # Save checkpoints on primary rank only
        if rank == 0 and val_loss < best_loss:
            best_loss = val_loss

            save_dir = Path(checkpoint_dir.best_dir if hasattr(checkpoint_dir, "best_dir") else "checkpoints/best")
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / "generator_checkpoint.pth"

            # Unwrap module state_dict if using DDP wrapper
            raw_gen = generator.module if is_distributed else generator
            raw_disc = discriminator.module if is_distributed else discriminator

            save_checkpoint(
                path=ckpt_path,
                epoch=epoch + 1,
                generator=raw_gen,
                discriminator=raw_disc,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                scaler=scaler,
                best_metric=best_loss,
            )

            LOGGER.info("Best checkpoint saved at %s", ckpt_path)

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()