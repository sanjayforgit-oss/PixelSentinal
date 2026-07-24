"""Data augmentation module."""

import numpy as np


class ImageAugmentor:
    """Apply augmentations to images."""
    
    def __init__(self, probability: float = 0.5):
        """Initialize augmentor.
        
        Args:
            probability: Probability of applying augmentation
        """
        self.probability = probability
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        return image