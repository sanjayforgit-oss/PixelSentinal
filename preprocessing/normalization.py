"""Normalization module for image preprocessing."""

import numpy as np
from typing import Tuple


class Normalizer:
    """Normalize image data."""
    
    def __init__(self, method: str = 'minmax'):
        """Initialize normalizer.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'percentile')
        """
        self.method = method
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if self.method == 'minmax':
            return (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.method == 'zscore':
            return (image - image.mean()) / (image.std() + 1e-8)
        return image