"""
===============================================================================
PixelSentinel - Utility Functions
-------------------------------------------------------------------------------
Author      : PixelSentinel Team
Project     : ISRO Bharatiya Antariksh Hackathon 2026

Purpose:
    Common helper functions used throughout the preprocessing pipeline.

Contains:
    - Directory creation
    - Earth Engine initialization
    - Landsat collection merging
    - Date generation
    - Safe filename generation
    - Metadata helpers
===============================================================================
"""

from pathlib import Path
from datetime import datetime
import ee

from preprocessing.config import (
    PROJECT_NAME,
    LANDSAT_COLLECTIONS,
)

from preprocessing.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Earth Engine Initialization
# =============================================================================

def initialize_gee() -> None:
    """
    Initialize Google Earth Engine.

    Raises
    ------
    RuntimeError
        If authentication fails.
    """

    try:
        ee.Initialize(project=PROJECT_NAME)
        logger.info("Google Earth Engine initialized successfully.")

    except Exception as e:
        logger.error("Failed to initialize Google Earth Engine.")
        raise RuntimeError(e)


# =============================================================================
# Merge Landsat Collections
# =============================================================================

def get_landsat_collection() -> ee.ImageCollection:
    """
    Merge Landsat 8 and Landsat 9 collections.

    Returns
    -------
    ee.ImageCollection
    """

    logger.info("Loading Landsat collections...")

    landsat8 = ee.ImageCollection(LANDSAT_COLLECTIONS[0])
    landsat9 = ee.ImageCollection(LANDSAT_COLLECTIONS[1])

    merged = landsat8.merge(landsat9)

    logger.info("Merged Landsat 8 + Landsat 9 collections.")

    return merged


# =============================================================================
# Create Directory
# =============================================================================

def create_directory(path: Path) -> None:
    """
    Create directory if it doesn't exist.

    Parameters
    ----------
    path : Path
    """

    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Create Region Directory
# =============================================================================

def create_region_directory(base_path: Path, region_name: str) -> Path:
    """
    Create a directory for a region.

    Example

    datasets/raw/Chennai

    Returns
    -------
    Path
    """

    region_path = base_path / region_name

    create_directory(region_path)

    return region_path


# =============================================================================
# Create Season Directory
# =============================================================================

def create_season_directory(region_path: Path, season: str) -> Path:
    """
    Create season directory.

    Example

    Chennai/
        Summer/

    Returns
    -------
    Path
    """

    season_path = region_path / season

    create_directory(season_path)

    return season_path


# =============================================================================
# Build Date Range
# =============================================================================

def build_date_range(year: int, season: tuple[str, str]) -> tuple[str, str]:
    """
    Convert season definition into full dates.

    Example

    2025 + ("-04-01","-06-15")

    →

    ("2025-04-01","2025-06-15")
    """

    start = f"{year}{season[0]}"
    end = f"{year}{season[1]}"

    return start, end


# =============================================================================
# Safe Scene Filename
# =============================================================================

def generate_scene_filename(
    region: str,
    season: str,
    image_id: str
) -> str:
    """
    Generate a readable filename.

    Example

    Chennai_Summer_LC09_145051_20250410.tif
    """

    image_id = image_id.replace("/", "_")

    return f"{region}_{season}_{image_id}.tif"


# =============================================================================
# Extract Satellite Name
# =============================================================================

def get_satellite_name(image_id: str) -> str:
    """
    Determine satellite from image id.

    Returns

    Landsat8

    Landsat9
    """

    if image_id.startswith("LC08"):
        return "Landsat8"

    if image_id.startswith("LC09"):
        return "Landsat9"

    return "Unknown"


# =============================================================================
# Get Current Timestamp
# =============================================================================

def current_timestamp() -> str:
    """
    Current timestamp.

    Example

    2026-07-17_20-35-10
    """

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# =============================================================================
# Scene Information
# =============================================================================

def print_scene_information(
    scene_id: str,
    region: str,
    season: str,
    cloud_cover: float,
) -> None:
    """
    Nicely log scene information.
    """

    logger.info("=" * 70)

    logger.info(f"Scene ID      : {scene_id}")

    logger.info(f"Region        : {region}")

    logger.info(f"Season        : {season}")

    logger.info(f"Cloud Cover   : {cloud_cover:.2f}%")

    logger.info("=" * 70)


# =============================================================================
# Verify Earth Engine Image
# =============================================================================

def verify_image(image: ee.Image) -> bool:
    """
    Check whether an image exists.

    Returns
    -------
    bool
    """

    try:

        image.bandNames().getInfo()

        return True

    except Exception:

        return False


# =============================================================================
# Pretty Divider
# =============================================================================

def divider() -> None:
    """
    Print separator in log.
    """

    logger.info("-" * 80)
    