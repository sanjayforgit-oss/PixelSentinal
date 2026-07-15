"""Colorization service."""

import torch
import numpy as np


class ColorizationService:
    """Service for image colorization."""
    
    def __init__(self, model_path: str):
        """Initialize service.
        
        Args:
            model_path: Path to model checkpoint
        """
        self.model_path = model_path
    
    def colorize(self, image: np.ndarray) -> np.ndarray:
        """Colorize image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Colorized RGB image
        """
        return image