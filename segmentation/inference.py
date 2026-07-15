"""Inference script for segmentation models."""

import torch
import numpy as np
from typing import Union


def segment_image(
    model,
    image: Union[torch.Tensor, np.ndarray],
    device: str = 'cuda'
) -> np.ndarray:
    """Segment image using model.
    
    Args:
        model: Segmentation model
        image: Input image
        device: Device to use
        
    Returns:
        Segmentation map
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device)
        output = model(image.unsqueeze(0))
    
    return output.argmax(1).squeeze(0).cpu().numpy()