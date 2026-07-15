"""Structural Similarity Index (SSIM) calculation."""

import numpy as np


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate SSIM between two images.
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        
    Returns:
        SSIM value
    """
    from skimage.metrics import structural_similarity as ssim
    return ssim(original, reconstructed, data_range=original.max() - original.min())