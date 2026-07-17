"""
utils.py

Utility functions used throughout the preprocessing pipeline.

Author: PixelSentinel Team
"""

from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import rasterio

from preprocessing.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------
# Directory Utilities
# ---------------------------------------------------------

def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def list_geotiff_files(directory: Path) -> List[Path]:
    """
    Returns all GeoTIFF files inside a directory.

    Parameters
    ----------
    directory : Path

    Returns
    -------
    List[Path]
    """
    return sorted(directory.glob("*.tif"))


# ---------------------------------------------------------
# Raster Utilities
# ---------------------------------------------------------

def read_geotiff(path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF.

    Returns
    -------
    image : np.ndarray
        Shape = (Bands, Height, Width)

    metadata : dict
    """
    logger.info(f"Reading {path.name}")

    with rasterio.open(path) as src:
        image = src.read()
        metadata = src.meta.copy()

    return image, metadata


def save_numpy(path: Path, array: np.ndarray) -> None:
    """
    Save NumPy array.
    """
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    """
    Load NumPy array.
    """
    return np.load(path)


# ---------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------

def normalize_band(
    band: np.ndarray,
    minimum: float = None,
    maximum: float = None,
) -> np.ndarray:
    """
    Normalize a band to [0,1].
    """

    band = band.astype(np.float32)

    if minimum is None:
        minimum = band.min()

    if maximum is None:
        maximum = band.max()

    if maximum == minimum:
        return np.zeros_like(band)

    return (band - minimum) / (maximum - minimum)


def normalize_multiband(image: np.ndarray) -> np.ndarray:
    """
    Normalize every band independently.

    Input
    -----
    (Bands,H,W)

    Output
    ------
    float32
    """

    normalized = []

    for band in image:
        normalized.append(normalize_band(band))

    return np.stack(normalized).astype(np.float32)


# ---------------------------------------------------------
# Dataset Utilities
# ---------------------------------------------------------

def train_val_test_split(
    file_list: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split files into train, validation and test.
    """

    random.seed(seed)

    file_list = file_list.copy()
    random.shuffle(file_list)

    n = len(file_list)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train = file_list[:train_end]
    val = file_list[train_end:val_end]
    test = file_list[val_end:]

    logger.info(
        f"Dataset Split -> Train:{len(train)} Val:{len(val)} Test:{len(test)}"
    )

    return train, val, test


# ---------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------

def verify_shape(image: np.ndarray, expected_channels: int) -> bool:
    """
    Verify number of channels.
    """

    if image.shape[0] != expected_channels:
        logger.warning(
            f"Expected {expected_channels} channels but found {image.shape[0]}"
        )
        return False

    return True


def print_metadata(metadata: dict) -> None:
    """
    Pretty-print raster metadata.
    """

    logger.info("Raster Metadata")

    for key, value in metadata.items():
        logger.info(f"{key}: {value}")
