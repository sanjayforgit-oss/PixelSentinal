"""
PixelSentinel - PyTorch Dataset & DataLoader Module (Local Training)
Handles reading multi-channel GeoTIFF tiles and executing synchronous 
on-the-fly Albumentations geometric transformations with proper [-1, 1] normalization.
"""
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


def get_transforms(is_train: bool = True) -> A.Compose:
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ], additional_targets={'target': 'image'})
    return A.Compose([], additional_targets={'target': 'image'})


class PixelSentinelDataset(Dataset):
    def __init__(self, split_dir: str, is_train: bool = True, transform: Optional[A.Compose] = None):
        self.split_dir = Path(split_dir)
        self.input_dir = self.split_dir / "input"
        self.target_dir = self.split_dir / "target"

        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(f"Missing 'input' or 'target' in {split_dir}")

        self.input_files = sorted(list(self.input_dir.glob("*.tif")) + list(self.input_dir.glob("*.tiff")))
        if len(self.input_files) == 0:
            raise RuntimeError(f"No GeoTIFF files found in {self.input_dir}")

        self.transform = transform if transform is not None else get_transforms(is_train)

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_path = self.input_files[idx]
        
        target_name = input_path.name.replace("_input.tif", "_target.tif").replace("_input.tiff", "_target.tiff")
        target_path = self.target_dir / target_name

        if not target_path.exists():
            target_path = self.target_dir / input_path.name

        with rasterio.open(input_path) as src_in:
            input_img = src_in.read().astype(np.float32).transpose(1, 2, 0)
        with rasterio.open(target_path) as src_target:
            target_img = src_target.read().astype(np.float32).transpose(1, 2, 0)

        # Apply Albumentations spatial transforms
        augmented = self.transform(image=input_img, target=target_img)
        
        input_np = augmented['image']
        target_np = augmented['target']

        # -------------------------------------------------------------
        # DYNAMIC MIN-MAX RANGE NORMALIZATION TO [-1.0, 1.0]
        # -------------------------------------------------------------
        # Safely scale target_np to [-1.0, 1.0]
        t_min, t_max = target_np.min(), target_np.max()
        if t_max - t_min > 1e-5:
            target_np = 2.0 * (target_np - t_min) / (t_max - t_min) - 1.0
        else:
            target_np = np.zeros_like(target_np)

        # Safely scale input_np to [-1.0, 1.0]
        i_min, i_max = input_np.min(), input_np.max()
        if i_max - i_min > 1e-5:
            input_np = 2.0 * (input_np - i_min) / (i_max - i_min) - 1.0
        else:
            input_np = np.zeros_like(input_np)

        # Convert to PyTorch Tensors (C, H, W)
        input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).float().contiguous()
        target_tensor = torch.from_numpy(target_np).permute(2, 0, 1).float().contiguous()

        return input_tensor, target_tensor


def get_dataloaders(
    root_dir: str = "datasets",
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    base_path = Path(root_dir)

    train_ds = PixelSentinelDataset(str(base_path / "train"), is_train=True)
    val_ds = PixelSentinelDataset(str(base_path / "val"), is_train=False)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    }