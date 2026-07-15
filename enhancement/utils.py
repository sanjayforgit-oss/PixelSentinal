"""Utility functions for enhancement module."""

import torch
import numpy as np
from typing import Union


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to image array.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Image array
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    return (tensor.numpy() * 255).astype(np.uint8)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert image array to tensor.
    
    Args:
        image: Input image
        
    Returns:
        Tensor
    """
    return torch.from_numpy(image.astype(np.float32) / 255.0)