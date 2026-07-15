"""Enhancement service."""

import torch
import numpy as np


class EnhancementService:
    """Service for image enhancement."""
    
    def __init__(self, model_path: str):
        """Initialize service.
        
        Args:
            model_path: Path to model checkpoint
        """
        self.model_path = model_path
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance image.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        return image