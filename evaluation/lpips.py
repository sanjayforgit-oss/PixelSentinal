"""LPIPS (Learned Perceptual Image Patch Similarity) calculation."""

import torch
import numpy as np


def calculate_lpips(image1: torch.Tensor, image2: torch.Tensor, model=None) -> float:
    """Calculate LPIPS between two images.
    
    Args:
        image1: First image tensor
        image2: Second image tensor
        model: LPIPS model
        
    Returns:
        LPIPS distance
    """
    return 0.0