"""
PixelSentinel Configuration File

This file contains all configurable parameters used throughout the
dataset collection and preprocessing pipeline.
"""

from pathlib import Path

# ==========================================================
# PROJECT DIRECTORIES
# ==========================================================

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Directories
DATASET_DIR = PROJECT_ROOT / "datasets"

RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Create directories automatically
for directory in [
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# GOOGLE EARTH ENGINE
# ==========================================================

GEE_PROJECT = "pixelsentinel"


# ==========================================================
# LANDSAT COLLECTION
# ==========================================================

LANDSAT_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

START_DATE = "2024-01-01"
END_DATE = "2026-06-01"

MAX_CLOUD_COVER = 5


# ==========================================================
# EXPORT SETTINGS
# ==========================================================

EXPORT_SCALE = 30          # meters/pixel
EXPORT_FORMAT = "GEO_TIFF"


# ==========================================================
# INPUT & TARGET BANDS
# ==========================================================

INPUT_BANDS = [
    "SR_B5",   # Near Infrared
    "SR_B6",   # SWIR-1
    "SR_B7",   # SWIR-2
    "ST_B10",  # Thermal Infrared
]

TARGET_BANDS = [
    "SR_B4",   # Red
    "SR_B3",   # Green
    "SR_B2",   # Blue
]


# ==========================================================
# IMAGE PROCESSING
# ==========================================================

TILE_SIZE = 256

NORMALIZATION_RANGE = (0.0, 1.0)


# ==========================================================
# REGIONS OF INTEREST (ROI)
# ==========================================================

ROI_LIST = {

    "Chennai": [79.95, 12.95, 80.25, 13.15],

    "Bengaluru": [77.35, 12.80, 77.85, 13.25],

    "Hyderabad": [78.20, 17.20, 78.80, 17.70],

    "Delhi": [76.90, 28.40, 77.40, 28.90],

    "Mumbai": [72.70, 18.80, 73.20, 19.40],
}