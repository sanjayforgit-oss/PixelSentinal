"""Loss functions for colorization."""

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """Perceptual loss for colorization."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        """Calculate perceptual loss.
        
        Args:
            x: Predicted output
            y: Target output
            
        Returns:
            Loss value
        """
        return torch.nn.functional.l1_loss(x, y)