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

METADATA_DIR = DATASET_DIR / "metadata"


TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Create directories automatically
for directory in [
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    METADATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Metadata File
METADATA_FILE = METADATA_DIR / "scene_metadata.csv"
# ==========================================================
# GOOGLE EARTH ENGINE
# ==========================================================

GEE_PROJECT = "pixelsentinel"


# ==========================================================
# LANDSAT COLLECTION
# ==========================================================

LANDSAT_COLLECTIONS = [
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
]

#YEARS = [2023,2024,]
YEARS = [2025,2026,]

SEASONS = {
    "Winter": ("-01-01", "-02-28"),
    "Summer": ("-04-01", "-06-15"),
    "Monsoon": ("-07-01", "-09-15"),
    "PostMonsoon": ("-10-01", "-12-15"),
}

MAX_SCENES_PER_SEASON = 3
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
# ==========================================================
# REGIONS OF INTEREST (ROI)
# Small bounding boxes to stay within GEE export limits
# ==========================================================

ROI_LIST = {

    # ======================================================
    # INDUSTRIAL AREAS
    # ======================================================

    "Jamshedpur": [86.10, 22.70, 86.40, 22.90],

    "Hazira": [72.60, 21.05, 72.90, 21.25],

    "VizagIndustrial": [83.15, 17.60, 83.45, 17.80],


    # ======================================================
    # AIRPORTS
    # ======================================================

    "ChennaiAirport": [80.12, 12.96, 80.22, 13.06],

    "DelhiAirport": [77.05, 28.52, 77.18, 28.62],

    "BengaluruAirport": [77.65, 13.15, 77.78, 13.28],


    # ======================================================
    # PORTS
    # ======================================================

    "ChennaiPort": [80.27, 13.08, 80.34, 13.15],

    "MumbaiPort": [72.82, 18.90, 72.90, 18.98],

    "MundraPort": [69.65, 22.72, 69.78, 22.85],

}

