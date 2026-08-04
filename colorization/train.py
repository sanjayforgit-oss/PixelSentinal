"""
PixelSentinel - Main Training Execution Script (Local Edition)
Enforces chunked training (e.g., 25 epochs per run) to protect laptop hardware.
Auto-resumes from the latest checkpoint and includes AC power failsafe protection.
"""
import logging
from pathlib import Path
import torch
import torch.optim as optim

from colorization.dataset import get_dataloaders
from colorization.models.discriminator import PatchGANDiscriminator
from colorization.models.generator import Generator
from colorization.models.weights import initialize_weights
from colorization.training.amp import create_grad_scaler
from colorization.training.checkpoint import load_checkpoint, save_checkpoint
from colorization.training.trainer import Pix2PixTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

EPOCHS_PER_RUN = 0
CHECKPOINT_PATH = Path("checkpoints/latest_checkpoint.pth")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Initializing Local Training on: {device}")

    # 1. Initialize Models
    generator = Generator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # 2. Initialize Optimizers & AMP Scaler
    gen_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
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
            device=device
        )
        start_epoch = checkpoint_data["epoch"] + 1

        for param_group in disc_opt.param_groups:
            param_group['lr'] = 0.00005
        LOGGER.info("Successfully updated Discriminator learning rate to 0.00005!")
    else:
        LOGGER.info("No checkpoint found. Initializing weights from scratch...")
        initialize_weights(generator)
        initialize_weights(discriminator)

    target_epoch = start_epoch + EPOCHS_PER_RUN - 1

    # 4. Load Data
    dataloaders = get_dataloaders(batch_size=8, num_workers=4)  # 8GB VRAM Safe
    trainer = Pix2PixTrainer(generator, discriminator, gen_opt, disc_opt, scaler, device)

    # 5. The Training Loop (Chunked)
    LOGGER.info(f"Starting chunked training run: Epoch {start_epoch} to {target_epoch}")
    
    for epoch in range(start_epoch, target_epoch + 1):
        LOGGER.info(f"\n--- Epoch {epoch} ---")
        
        try:
            # Train one epoch
            train_metrics = trainer.train_epoch(dataloaders["train"], epoch)
            LOGGER.info(
                f"Epoch {epoch} Completed | "
                f"Avg G_Loss: {train_metrics['g_loss']:.4f} | "
                f"Avg D_Loss: {train_metrics['d_loss']:.4f}"
            )

            # Save Checkpoint at the end of every epoch
            save_checkpoint(
                path=CHECKPOINT_PATH,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=gen_opt,
                discriminator_optimizer=disc_opt,
                scaler=scaler
            )

        except RuntimeError as e:
            if "ChargerDisconnected" in str(e):
                LOGGER.warning("🛑 Emergency shutdown triggered by unplugged charger!")
                LOGGER.info(f"Progress through Epoch {epoch - 1} remains safely stored in {CHECKPOINT_PATH}.")
                return  # Exit process safely

    LOGGER.info(f"\n✅ Training block complete! Reached Epoch {target_epoch}.")


if __name__ == "__main__":
    main()