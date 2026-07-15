"""Normalized Difference Water Index (NDWI) calculation."""

import numpy as np


def calculate_ndwi(nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
    """Calculate NDWI from NIR and SWIR bands.
    
    Args:
        nir_band: Near-infrared band data
        swir_band: Short-wave infrared band data
        
    Returns:
        NDWI values
    """
    ndwi = (nir_band - swir_band) / (nir_band + swir_band + 1e-8)
    return ndwi