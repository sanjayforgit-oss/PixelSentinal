"""Peak Signal-to-Noise Ratio (PSNR) calculation."""

import numpy as np


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: int = 255) -> float:
    """Calculate PSNR between two images.
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        max_val: Maximum pixel value
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))