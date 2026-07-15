"""Detection service."""

import numpy as np
from typing import List, Dict


class DetectionService:
    """Service for object detection."""
    
    def __init__(self, model_path: str):
        """Initialize service.
        
        Args:
            model_path: Path to model checkpoint
        """
        self.model_path = model_path
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in image.
        
        Args:
            image: Input image
            
        Returns:
            List of detections
        """
        return []