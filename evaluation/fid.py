"""Fréchet Inception Distance (FID) calculation."""

import numpy as np


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Calculate FID between real and fake image features.
    
    Args:
        real_features: Features from real images
        fake_features: Features from fake images
        
    Returns:
        FID score
    """
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    cov_real = np.cov(real_features.T)
    cov_fake = np.cov(fake_features.T)
    
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    return mean_diff