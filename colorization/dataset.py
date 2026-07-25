"""
PixelSentinel - PyTorch Dataset & DataLoader Module (DDP Ready)

Handles reading multi-channel GeoTIFF tiles (4-channel input, 3-channel target),
executing synchronous on-the-fly Albumentations geometric transformations,
and configuring DataLoaders with DistributedSampler for multi-node training.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import albumentations as A


def get_transforms(is_train: bool = True) -> A.Compose:
    """
    Constructs Albumentations pipelines for paired spatial augmentations.
    Only geometric transformations are used to preserve spectral relationships.
    """
    if is_train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={'target': 'image'}
        )
    else:
        return A.Compose([], additional_targets={'target': 'image'})


class PixelSentinelDataset(Dataset):
    """
    PyTorch Dataset for PixelSentinel paired Landsat tiles.
    Reads input (4 channels: B5, B6, B7, B10) and target (3 channels: B4, B3, B2).
    """
    def __init__(self, split_dir: str, is_train: bool = True, transform: Optional[A.Compose] = None):
        self.split_dir = Path(split_dir)
        self.input_dir = self.split_dir / "input"
        self.target_dir = self.split_dir / "target"

        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(f"Invalid dataset structure. Missing 'input' or 'target' in {split_dir}")

        self.input_files = sorted(list(self.input_dir.glob("*.tif")) + list(self.input_dir.glob("*.tiff")))

        if len(self.input_files) == 0:
            raise RuntimeError(f"No GeoTIFF files found in {self.input_dir}")

        self.transform = transform if transform is not None else get_transforms(is_train)

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_path = self.input_files[idx]
        target_path = self.target_dir / input_path.name

        if not target_path.exists():
            raise FileNotFoundError(f"Missing corresponding target tile: {target_path}")

        with rasterio.open(input_path) as src_in:
            input_img = src_in.read().astype(np.float32)
            input_img = np.transpose(input_img, (1, 2, 0))

        with rasterio.open(target_path) as src_target:
            target_img = src_target.read().astype(np.float32)
            target_img = np.transpose(target_img, (1, 2, 0))

        # Synchronous geometric transformations
        augmented = self.transform(image=input_img, target=target_img)
        aug_input = augmented['image']
        aug_target = augmented['target']

        input_tensor = torch.from_numpy(aug_input).permute(2, 0, 1).contiguous()
        target_tensor = torch.from_numpy(aug_target).permute(2, 0, 1).contiguous()

        return input_tensor, target_tensor


def get_dataloaders(
    root_dir: str = "datasets",
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    is_distributed: bool = False
) -> Dict[str, DataLoader]:
    """
    Factory helper function to initialize PyTorch DataLoaders.
    Includes support for Distributed Data Parallel (DDP).
    """
    base_path = Path(root_dir)

    train_dataset = PixelSentinelDataset(split_dir=str(base_path / "train"), is_train=True)
    val_dataset = PixelSentinelDataset(split_dir=str(base_path / "val"), is_train=False)
    test_dataset = PixelSentinelDataset(split_dir=str(base_path / "test"), is_train=False)

    # DDP Samplers
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    # In DDP, the sampler handles shuffling, so shuffle must be False if sampler is used
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None), 
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    }

    return dataloaders