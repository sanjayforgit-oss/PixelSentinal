"""
PixelSentinel - Main Training Execution Script (Local Edition)
Enforces chunked training (e.g., 25 epochs per run) to protect laptop hardware.
Auto-resumes from the latest checkpoint and includes AC power failsafe protection.
Includes Total Variation (TV) loss to suppress high-frequency grid artifacts.
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from colorization.dataset import get_dataloaders
from colorization.models.discriminator import PatchGANDiscriminator
from colorization.models.generator import Generator
from colorization.models.weights import initialize_weights
from colorization.training.amp import create_grad_scaler
from colorization.training.checkpoint import load_checkpoint, save_checkpoint
from colorization.training.trainer import Pix2PixTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# ==========================================================
# ⚙️ CONFIGURATION PARAMETERS
# ==========================================================
EPOCHS_PER_RUN = 15
CHECKPOINT_PATH = Path("checkpoints/latest_checkpoint.pth")
LAMBDA_TV = 10.0  # Total Variation loss weight to suppress grid mesh lines
GEN_LR = 0.0002   # Generator initial LR
DISC_LR = 0.00005 # Reduced Discriminator LR (TTUR) to prevent grid forcing
# ==========================================================


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Penalizes high-frequency grid noise and pixel-to-pixel jump artifacts
    by calculating spatial gradients across adjacent image pixels.
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Initializing Local Training on: {device}")

    # 1. Initialize Models
    generator = Generator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # 2. Initialize Optimizers & AMP Scaler
    gen_opt = optim.Adam(generator.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    disc_opt = optim.Adam(discriminator.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
    scaler = create_grad_scaler()

    # 3. Load Checkpoint or Initialize Fresh Weights
    start_epoch = 1
    if CHECKPOINT_PATH.exists():
        LOGGER.info(f"Found existing checkpoint at {CHECKPOINT_PATH}. Resuming...")
        checkpoint_data = load_checkpoint(
            path=CHECKPOINT_PATH,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=gen_opt,
            discriminator_optimizer=disc_opt,
            scaler=scaler,
            device=device,
        )
        start_epoch = checkpoint_data["epoch"] + 1

        # Enforce Two-Time Scale Update Rule (TTUR): Keep Discriminator LR lowered
        for param_group in disc_opt.param_groups:
            param_group["lr"] = DISC_LR
        LOGGER.info(f"Successfully set Discriminator learning rate to {DISC_LR} (TTUR enabled).")
    else:
        LOGGER.info("No checkpoint found. Initializing weights from scratch...")
        initialize_weights(generator)
        initialize_weights(discriminator)

    target_epoch = start_epoch + EPOCHS_PER_RUN - 1

    # 4. Learning Rate Cosine Scheduler (Smooth decay over 200 epochs)
    scheduler_g = CosineAnnealingLR(gen_opt, T_max=200, eta_min=1e-5)
    scheduler_d = CosineAnnealingLR(disc_opt, T_max=200, eta_min=1e-5)

    # Fast-forward scheduler steps if resuming from a checkpoint
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler_g.step()
            scheduler_d.step()

    # 5. Load Data & Initialize Trainer
    dataloaders = get_dataloaders(batch_size=8, num_workers=4)  # 8GB VRAM Safe
    trainer = Pix2PixTrainer(generator, discriminator, gen_opt, disc_opt, scaler, device)

    # Inject TV Loss function & weight into trainer instance
    if hasattr(trainer, "lambda_tv"):
        trainer.lambda_tv = LAMBDA_TV
    if hasattr(trainer, "tv_loss_fn"):
        trainer.tv_loss_fn = total_variation_loss

    # 6. The Chunked Training Loop
    LOGGER.info(f"Starting chunked training run: Epoch {start_epoch} to {target_epoch}")
    LOGGER.info(f"Grid suppression active | TV Loss Weight (\u03bb_tv): {LAMBDA_TV}")

    for epoch in range(start_epoch, target_epoch + 1):
        LOGGER.info(f"\n--- Epoch {epoch} ---")

        try:
            # Train one epoch
            train_metrics = trainer.train_epoch(dataloaders["train"], epoch)
            
            # Step learning rate schedulers
            scheduler_g.step()
            scheduler_d.step()
            
            current_g_lr = gen_opt.param_groups[0]["lr"]
            LOGGER.info(
                f"Epoch {epoch} Completed | "
                f"Avg G_Loss: {train_metrics['g_loss']:.4f} | "
                f"Avg D_Loss: {train_metrics['d_loss']:.4f} | "
                f"G_LR: {current_g_lr:.6f}"
            )

            # Save Checkpoint at the end of every epoch
            save_checkpoint(
                path=CHECKPOINT_PATH,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=gen_opt,
                discriminator_optimizer=disc_opt,
                scaler=scaler,
            )

        except RuntimeError as e:
            if "ChargerDisconnected" in str(e):
                LOGGER.warning("🛑 Emergency shutdown triggered by unplugged charger!")
                LOGGER.info(f"Progress through Epoch {epoch - 1} remains safely stored in {CHECKPOINT_PATH}.")
                return  # Exit process safely
            raise e

    LOGGER.info(f"\n✅ Training block complete! Reached Epoch {target_epoch}.")


if __name__ == "__main__":
    main()