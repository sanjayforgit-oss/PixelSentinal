"""Inference script for colorization models."""

import torch
import numpy as np
from typing import Union


def colorize_image(
    model,
    image: Union[torch.Tensor, np.ndarray],
    device: str = 'cuda'
) -> np.ndarray:
    """Colorize infrared image.
    
    Args:
        model: Colorization model
        image: Input grayscale image
        device: Device to use
        
    Returns:
        Colorized RGB image
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device)
        output = model(image.unsqueeze(0))
    
    return output.squeeze(0).cpu().numpy()