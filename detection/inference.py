"""Inference script for object detection."""

import torch
import numpy as np
from typing import List, Dict


def detect_objects(
    model,
    image: np.ndarray,
    confidence_threshold: float = 0.5,
    device: str = 'cuda'
) -> List[Dict]:
    """Detect objects in image.
    
    Args:
        model: Detection model
        image: Input image
        confidence_threshold: Confidence threshold
        device: Device to use
        
    Returns:
        List of detections
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device)
        detections = model(image.unsqueeze(0))
    
    return []