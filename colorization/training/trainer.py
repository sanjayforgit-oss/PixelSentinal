"""
Trainer for PixelSentinel Pix2Pix.

This module implements the training pipeline for the Pix2Pix model.
It orchestrates the generator, discriminator, optimizers, automatic
mixed precision, loss computation, checkpointing, and validation.

The trainer intentionally contains no dataset-specific preprocessing.
"""

from __future__ import annotations

import logging
#from pathlib import Path
#from typing import Any

import torch
#import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from configs.config import CONFIG
from colorization.models.discriminator import PatchGANDiscriminator
from colorization.models.generator import Generator
from colorization.training.losses import Pix2PixLoss

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Pix2PixTrainer",
]


class Pix2PixTrainer:
    """
    Trainer for Pix2Pix.

    Responsibilities
    ----------------
    - Model orchestration
    - One training iteration
    - One validation iteration
    - Optimizer updates
    - Mixed precision compatibility
    - Metric aggregation

    Checkpointing is intentionally delegated to checkpoint.py.
    """

    def __init__(
        self,
        generator: Generator,
        discriminator: PatchGANDiscriminator,
        generator_optimizer: Optimizer,
        discriminator_optimizer: Optimizer,
        device: torch.device,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        """
        Initialize trainer.

        Parameters
        ----------
        generator:
            Generator model.

        discriminator:
            PatchGAN discriminator.

        generator_optimizer:
            Optimizer for generator.

        discriminator_optimizer:
            Optimizer for discriminator.

        device:
            Training device.

        scaler:
            AMP gradient scaler.
        """
        self.device = device

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.loss_fn = Pix2PixLoss().to(device)

        self.scaler = scaler

        self.use_amp = bool(
            getattr(CONFIG.training, "mixed_precision", False)
        )

    @staticmethod
    def _move_to_device(
        *tensors: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, ...]:
        """
        Move tensors to the configured device.
        """
        return tuple(
            tensor.to(device, non_blocking=True)
            for tensor in tensors
        )
    
    def train_step(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        """
        Execute a single training step.

        Parameters
        ----------
        inputs:
            Input infrared images.

        targets:
            Ground-truth RGB images.

        Returns
        -------
        dict[str, float]
            Dictionary containing scalar loss values.
        """

        self.generator.train()
        self.discriminator.train()

        inputs, targets = self._move_to_device(
            inputs,
            targets,
            device=self.device,
        )

        amp_device = (
            "cuda"
            if self.device.type == "cuda"
            else "cpu"
        )

        #
        # =========================================================
        # 1. Train Discriminator
        # =========================================================
        #

        self.discriminator_optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=amp_device,
            enabled=self.use_amp,
        ):

            with torch.no_grad():
                generated = self.generator(inputs)

            real_prediction = self.discriminator(
                inputs,
                targets,
            )

            fake_prediction = self.discriminator(
                inputs,
                generated,
            )

            discriminator_loss, real_loss, fake_loss = (
                self.loss_fn.discriminator_loss(
                    real_prediction=real_prediction,
                    fake_prediction=fake_prediction,
                )
            )

        if self.scaler is not None and self.use_amp:

            self.scaler.scale(
                discriminator_loss
            ).backward()

            self.scaler.step(
                self.discriminator_optimizer
            )

        else:

            discriminator_loss.backward()

            self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad(set_to_none=True)


        #
        # =========================================================
        # 2. Train Generator
        # =========================================================
        #

        self.generator_optimizer.zero_grad(
            set_to_none=True
        )

        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)

        self.discriminator.eval()


        with torch.amp.autocast(
            device_type=amp_device,
            enabled=self.use_amp,
        ):

            generated = self.generator(inputs)

            fake_prediction = self.discriminator(
                inputs,
                generated,
            )

            generator_loss, gan_loss, l1_loss = (
                self.loss_fn.generator_loss(
                    fake_prediction=fake_prediction,
                    generated_image=generated,
                    target_image=targets,
                )
            )


        if self.scaler is not None and self.use_amp:

            self.scaler.scale(
                generator_loss
            ).backward()

            self.scaler.step(
                self.generator_optimizer
            )

            self.scaler.update()

        else:

            generator_loss.backward()

            self.generator_optimizer.step()


        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(True)
        self.discriminator.train()

        return {
            "generator_loss": float(
                generator_loss.detach().item()
            ),
            "generator_gan_loss": float(
                gan_loss.detach().item()
            ),
            "generator_l1_loss": float(
                l1_loss.detach().item()
            ),
            "discriminator_loss": float(
                discriminator_loss.detach().item()
            ),
            "discriminator_real_loss": float(
                real_loss.detach().item()
            ),
            "discriminator_fake_loss": float(
                fake_loss.detach().item()
            ),
        }
    
    @torch.no_grad()
    def validate_step(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        """
        Execute a single validation step.

        Parameters
        ----------
        inputs:
            Input infrared images.

        targets:
            Ground-truth RGB images.

        Returns
        -------
        dict[str, float]
            Dictionary containing validation losses.
        """
        self.generator.eval()
        self.discriminator.eval()

        inputs, targets = self._move_to_device(
            inputs,
            targets,
            device=self.device,
        )

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"

        with torch.amp.autocast(
            device_type=amp_device,
            enabled=self.use_amp,
        ):
            generated = self.generator(inputs)

            real_prediction = self.discriminator(
                inputs,
                targets,
            )

            fake_prediction = self.discriminator(
                inputs,
                generated,
            )

            losses = self.loss_fn(
                real_prediction=real_prediction,
                fake_prediction_for_discriminator=fake_prediction,
                fake_prediction_for_generator=fake_prediction,
                generated_image=generated,
                target_image=targets,
            )

        return losses.as_dict()

    def fit_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """
        Train the model for one epoch.

        Parameters
        ----------
        dataloader:
            Training dataloader.

        Returns
        -------
        dict[str, float]
            Mean losses over the epoch.
        """
        metrics: dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            inputs, targets = batch

            batch_metrics = self.train_step(
                inputs=inputs,
                targets=targets,
            )

            if not metrics:
                metrics = {
                    key: 0.0 for key in batch_metrics
                }

            for key, value in batch_metrics.items():
                metrics[key] += value

            num_batches += 1

        if num_batches == 0:
            return metrics

        return {
            key: value / num_batches
            for key, value in metrics.items()
        }

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """
        Validate the model for one epoch.

        Parameters
        ----------
        dataloader:
            Validation dataloader.

        Returns
        -------
        dict[str, float]
            Mean validation losses.
        """
        metrics: dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            inputs, targets = batch

            batch_metrics = self.validate_step(
                inputs=inputs,
                targets=targets,
            )

            if not metrics:
                metrics = {
                    key: 0.0 for key in batch_metrics
                }

            for key, value in batch_metrics.items():
                metrics[key] += value

            num_batches += 1

        if num_batches == 0:
            return metrics

        return {
            key: value / num_batches
            for key, value in metrics.items()
        }