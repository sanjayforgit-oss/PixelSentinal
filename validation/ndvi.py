"""Normalized Difference Vegetation Index (NDVI) calculation."""

import numpy as np


def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Calculate NDVI from red and NIR bands.
    
    Args:
        red_band: Red band data
        nir_band: Near-infrared band data
        
    Returns:
        NDVI values
    """
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    return ndvi