"""
===============================================================================
PixelSentinel - Configuration File
-------------------------------------------------------------------------------
Author      : PixelSentinel Team
Project     : ISRO Bharatiya Antariksh Hackathon 2026
Module      : Dataset Configuration

Purpose:
    Central configuration for dataset collection using Google Earth Engine.

Contains:
    - Landsat collections
    - Regions of Interest (ROIs)
    - Seasons
    - Band selection
    - Download settings
    - Directory paths
===============================================================================
"""

from pathlib import Path
import ee

# =============================================================================
# PROJECT INFORMATION
# =============================================================================

PROJECT_NAME = "pixelsentinel"

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "datasets"

RAW_DATASET_DIR = DATASET_DIR / "raw"

PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

METADATA_DIR = DATASET_DIR / "metadata"

LOG_DIR = ROOT_DIR / "logs"

# Create directories automatically

for directory in [
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    METADATA_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LANDSAT COLLECTIONS
# =============================================================================

LANDSAT_COLLECTIONS = [
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
]

# =============================================================================
# PIXELSENTINEL BAND CONFIGURATION
# =============================================================================

INPUT_BANDS = [
    "SR_B5",     # Near Infrared
    "SR_B6",     # SWIR-1
    "SR_B7",     # SWIR-2
    "ST_B10",    # Thermal Infrared
]

TARGET_BANDS = [
    "SR_B4",     # Red
    "SR_B3",     # Green
    "SR_B2",     # Blue
]

ALL_BANDS = INPUT_BANDS + TARGET_BANDS

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

EXPORT_SCALE = 30

EXPORT_FORMAT = "GEO_TIFF"

MAX_CLOUD_COVER = 5

MAX_SCENES_PER_SEASON = 2

# =============================================================================
# YEARS
# =============================================================================

YEARS = [
    2023,
    2024,
    2025,
    2026,
]

# =============================================================================
# SEASONS
# =============================================================================

SEASONS = {

    "Winter": (
        "-01-01",
        "-02-28"
    ),

    "Summer": (
        "-04-01",
        "-06-15"
    ),

    "Monsoon": (
        "-07-01",
        "-09-15"
    ),

    "PostMonsoon": (
        "-10-01",
        "-12-15"
    ),
}

# =============================================================================
# REGIONS OF INTEREST
# =============================================================================
#
# IMPORTANT:
#
# Each ROI is intentionally small enough to remain below the
# Google Earth Engine direct download limit (~48 MB).
#
# Every ROI represents a different land-cover type.
#
# =============================================================================

ROIS = {

    ###########################################################################
    # URBAN
    ###########################################################################

    "Chennai": ee.Geometry.Rectangle(
        [79.95, 12.95, 80.25, 13.15]
    ),

    "Bengaluru": ee.Geometry.Rectangle(
        [77.45, 12.85, 77.75, 13.15]
    ),

    "Hyderabad": ee.Geometry.Rectangle(
        [78.25, 17.25, 78.55, 17.55]
    ),

    "Delhi": ee.Geometry.Rectangle(
        [76.95, 28.45, 77.25, 28.75]
    ),

    "Mumbai": ee.Geometry.Rectangle(
        [72.75, 18.85, 73.05, 19.15]
    ),

    ###########################################################################
    # AGRICULTURE
    ###########################################################################

    "Punjab_Farmland": ee.Geometry.Rectangle(
        [75.10, 30.70, 75.40, 31.00]
    ),

    "TamilNadu_Farmland": ee.Geometry.Rectangle(
        [78.90, 10.80, 79.20, 11.10]
    ),

    ###########################################################################
    # FOREST
    ###########################################################################

    "WesternGhats": ee.Geometry.Rectangle(
        [76.10, 10.00, 76.40, 10.30]
    ),

    "Assam_Forest": ee.Geometry.Rectangle(
        [91.50, 26.00, 91.80, 26.30]
    ),

    ###########################################################################
    # WATER
    ###########################################################################

    "ChilikaLake": ee.Geometry.Rectangle(
        [85.15, 19.55, 85.45, 19.85]
    ),

    "Sundarbans": ee.Geometry.Rectangle(
        [88.50, 21.60, 88.80, 21.90]
    ),

    ###########################################################################
    # DESERT
    ###########################################################################

    "TharDesert": ee.Geometry.Rectangle(
        [70.90, 26.80, 71.20, 27.10]
    ),

    ###########################################################################
    # MOUNTAINS
    ###########################################################################

    "Himalayas": ee.Geometry.Rectangle(
        [77.00, 31.00, 77.30, 31.30]
    ),

}

# =============================================================================
# DOWNLOAD SETTINGS
# =============================================================================

DOWNLOAD_TIMEOUT = 120

DOWNLOAD_RETRIES = 3

# =============================================================================
# IMAGE QUALITY SETTINGS
# =============================================================================

MIN_VALID_PIXELS = 0.80

REMOVE_EMPTY_TILES = True

REMOVE_HIGH_CLOUD_TILES = True

# =============================================================================
# FILE NAMING
# =============================================================================

FILE_PREFIX = "pixelsentinel"

# =============================================================================
# METADATA FILE
# =============================================================================

METADATA_FILENAME = "scene_metadata.csv"

# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = LOG_DIR / "dataset_builder.log"

# =============================================================================
# DEBUG
# =============================================================================

DEBUG = True