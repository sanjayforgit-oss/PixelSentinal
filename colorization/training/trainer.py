"""
Training step logic for PixelSentinel Pix2Pix.
    Handles forward passes, loss computation, AMP optimization steps, and battery safety checks.
"""
import logging
from typing import Dict

import psutil
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from colorization.models.losses import Pix2PixLoss
from colorization.training.amp import autocast_context

LOGGER = logging.getLogger(__name__)


class Pix2PixTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_opt: Optimizer,
        disc_opt: Optimizer,
        scaler: torch.amp.GradScaler,
        device: torch.device
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.scaler = scaler
        self.device = device
        self.criterion = Pix2PixLoss().to(device)

    @staticmethod
    def check_power_status() -> bool:
        """Returns False if running on battery (charger unplugged)."""
        battery = psutil.sensors_battery()
        if battery is not None:
            return battery.power_plugged
        return True  # Returns True for desktop PCs or if sensor is unavailable

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {"g_loss": 0.0, "d_loss": 0.0}

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # --- AC POWER FAILSAFE ---
            if not self.check_power_status():
                LOGGER.warning("⚠️ CHARGER DISCONNECTED! Halting training to preserve battery power.")
                raise RuntimeError("ChargerDisconnected")

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.gen_opt.zero_grad()
            self.disc_opt.zero_grad()

            with autocast_context(self.device):
                # 1. Generate fake images
                fake_targets = self.generator(inputs)

                # 2. Discriminator predictions
                real_pred = self.discriminator(inputs, targets)
                fake_pred_d = self.discriminator(inputs, fake_targets.detach())
                fake_pred_g = self.discriminator(inputs, fake_targets)

                # 3. Calculate all losses using your losses.py
                losses = self.criterion(
                    real_prediction=real_pred,
                    fake_prediction_for_discriminator=fake_pred_d,
                    fake_prediction_for_generator=fake_pred_g,
                    generated_image=fake_targets,
                    target_image=targets
                )

            # 4. Backward & Optimize (Using AMP)
            self.scaler.scale(losses.discriminator_loss).backward(retain_graph=True)
            self.scaler.scale(losses.generator_loss).backward()

            self.scaler.step(self.disc_opt)
            self.scaler.step(self.gen_opt)
            self.scaler.update()

            epoch_losses["g_loss"] += losses.generator_loss.item()
            epoch_losses["d_loss"] += losses.discriminator_loss.item()

            if batch_idx % 50 == 0:
                LOGGER.info(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] - "
                            f"G_Loss: {losses.generator_loss.item():.4f} - "
                            f"D_Loss: {losses.discriminator_loss.item():.4f}")

        # Average losses
        return {k: v / len(dataloader) for k, v in epoch_losses.items()}