"""
PixelSentinel - Distributed Training Script (Elastic DDP Edition)
Wraps the existing train.py logic to support multi-node PyTorch Elastic Training.
Nodes can join mid-training as they finish copying files.

Run via torchrun on each node:
  torchrun --nnodes=1:10 --nproc-per-node=1 \
           --rdzv-backend=c10d \
           --rdzv-endpoint=10.0.4.247:29400 \
           --rdzv-id=pixelsentinel_train \
           -m colorization.train_ddp
"""
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from colorization.dataset import get_dataloaders
from colorization.models.discriminator import PatchGANDiscriminator
from colorization.models.generator import Generator
from colorization.models.weights import initialize_weights
from colorization.training.amp import create_grad_scaler
from colorization.training.checkpoint import load_checkpoint, save_checkpoint
from colorization.training.trainer import Pix2PixTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

EPOCHS_PER_RUN = 10
CHECKPOINT_PATH = Path("checkpoints/latest_checkpoint.pth")


def setup_ddp():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"[Rank {rank}/{world_size}] Initializing DDP Training on: {device}")

    # 1. Initialize Models & Weights
    generator = Generator().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    initialize_weights(generator)
    initialize_weights(discriminator)

    # Wrap with DDP
    generator = DDP(generator, device_ids=[local_rank] if torch.cuda.is_available() else None)
    discriminator = DDP(discriminator, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # 2. Optimizers & AMP Scaler
    gen_opt = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    disc_opt = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scaler = create_grad_scaler()

    # 3. Load Checkpoint (only rank 0 reads, then broadcasts)
    start_epoch = 1
    if rank == 0 and CHECKPOINT_PATH.exists():
        LOGGER.info(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming...")
        checkpoint_data = load_checkpoint(
            path=CHECKPOINT_PATH,
            generator=generator.module,
            discriminator=discriminator.module,
            generator_optimizer=gen_opt,
            discriminator_optimizer=disc_opt,
            scaler=scaler,
            device=device,
        )
        start_epoch = checkpoint_data["epoch"] + 1

    # Broadcast start_epoch to all ranks
    start_epoch_tensor = torch.tensor(start_epoch, dtype=torch.int32).to(device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())

    target_epoch = start_epoch + EPOCHS_PER_RUN - 1

    # 4. Distributed DataLoaders (each rank gets a different shard)
    dataloaders = get_dataloaders(
        batch_size=8,
        num_workers=4,
        distributed=True,      # Will use DistributedSampler if supported
        rank=rank,
        world_size=world_size,
    )

    trainer = Pix2PixTrainer(generator, discriminator, gen_opt, disc_opt, scaler, device)

    # 5. Training Loop
    if rank == 0:
        LOGGER.info(f"Starting DDP training: Epoch {start_epoch} to {target_epoch} | World size: {world_size}")

    for epoch in range(start_epoch, target_epoch + 1):
        # Shuffle sampler each epoch
        if hasattr(dataloaders["train"].sampler, "set_epoch"):
            dataloaders["train"].sampler.set_epoch(epoch)

        if rank == 0:
            LOGGER.info(f"\n--- Epoch {epoch} ---")

        try:
            train_metrics = trainer.train_epoch(dataloaders["train"], epoch)

            if rank == 0:
                LOGGER.info(
                    f"Epoch {epoch} | G_Loss: {train_metrics['g_loss']:.4f} | "
                    f"D_Loss: {train_metrics['d_loss']:.4f} | "
                    f"Nodes: {world_size}"
                )
                # Only rank 0 saves checkpoint
                save_checkpoint(
                    path=CHECKPOINT_PATH,
                    epoch=epoch,
                    generator=generator.module,
                    discriminator=discriminator.module,
                    generator_optimizer=gen_opt,
                    discriminator_optimizer=disc_opt,
                    scaler=scaler,
                )

        except RuntimeError as e:
            if rank == 0:
                LOGGER.error(f"Runtime error: {e}")
            break

        # Sync all ranks before next epoch
        dist.barrier()

    if rank == 0:
        LOGGER.info(f"\n✅ Training block complete! Reached Epoch {target_epoch}.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
