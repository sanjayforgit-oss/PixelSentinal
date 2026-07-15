"""Helper functions."""

import numpy as np
from typing import Union, Tuple


def normalize_image(image: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray:
    """Normalize image to range [min_val, max_val].
    
    Args:
        image: Input image
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Normalized image
    """
    img_min = image.min()
    img_max = image.max()
    normalized = (image - img_min) / (img_max - img_min + 1e-8)
    return normalized * (max_val - min_val) + min_val


def clip_image(image: np.ndarray, min_val: float = 0, max_val: float = 255) -> np.ndarray:
    """Clip image values to range.
    
    Args:
        image: Input image
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped image
    """
    return np.clip(image, min_val, max_val)