"""Band extraction module for satellite imagery."""

import numpy as np
from typing import List, Tuple


class BandExtractor:
    """Extract specific bands from satellite imagery."""
    
    def __init__(self, bands: List[int]):
        """Initialize band extractor.
        
        Args:
            bands: List of band indices to extract
        """
        self.bands = bands
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract specified bands from image.
        
        Args:
            image: Input image array
            
        Returns:
            Array with extracted bands
        """
        return image[self.bands, :, :]