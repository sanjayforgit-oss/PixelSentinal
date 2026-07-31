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
        
        epoch_losses = {
            "g_loss": 0.0,
            "d_loss": 0.0,
            "perc_loss": 0.0,
            "spec_loss": 0.0,
        }

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # --- AC POWER FAILSAFE ---
            if not self.check_power_status():
                LOGGER.warning("⚠️ CHARGER DISCONNECTED! Halting training to preserve battery power.")
                raise RuntimeError("ChargerDisconnected")

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # =========================================================
            # 1. Train Discriminator
            # =========================================================
            self.disc_opt.zero_grad()
            with autocast_context(self.device):
                with torch.no_grad():
                    fake_targets = self.generator(inputs)
                
                real_pred = self.discriminator(inputs, targets)
                fake_pred_d = self.discriminator(inputs, fake_targets.detach())
                
                d_loss, _, _ = self.criterion.discriminator_loss(real_pred, fake_pred_d)

            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.disc_opt)

            # =========================================================
            # 2. Train Generator
            # =========================================================
            self.gen_opt.zero_grad()
            with autocast_context(self.device):
                fake_targets_g = self.generator(inputs)
                fake_pred_g = self.discriminator(inputs, fake_targets_g)

                g_total_loss, g_gan_loss, g_l1_loss, g_perc_loss, g_spec_loss = self.criterion.generator_loss(
                    input_image=inputs,
                    fake_prediction=fake_pred_g,
                    generated_image=fake_targets_g,
                    target_image=targets,
                )

            self.scaler.scale(g_total_loss).backward()
            self.scaler.step(self.gen_opt)
            self.scaler.update()

            # Record losses
            epoch_losses["g_loss"] += g_total_loss.item()
            epoch_losses["d_loss"] += d_loss.item()
            epoch_losses["perc_loss"] += g_perc_loss.item()
            epoch_losses["spec_loss"] += g_spec_loss.item()

            if batch_idx % 50 == 0:
                LOGGER.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] - "
                    f"G_Loss: {g_total_loss.item():.4f} - "
                    f"D_Loss: {d_loss.item():.4f} - "
                    f"Perc_Loss: {g_perc_loss.item():.4f} - "
                    f"Spec_Loss: {g_spec_loss.item():.4f}"
                )

        # Average losses over all batches
        return {k: v / len(dataloader) for k, v in epoch_losses.items()}