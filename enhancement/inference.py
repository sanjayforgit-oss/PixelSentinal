"""Inference script for enhancement models."""

import torch
import numpy as np
from typing import Union


def enhance_image(
    model,
    image: Union[torch.Tensor, np.ndarray],
    device: str = 'cuda'
) -> np.ndarray:
    """Enhance image using model.
    
    Args:
        model: Enhancement model
        image: Input image
        device: Device to use
        
    Returns:
        Enhanced image
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device)
        output = model(image.unsqueeze(0))
    
    return output.squeeze(0).cpu().numpy()