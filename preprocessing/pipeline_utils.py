"""
PixelSentinel — Preprocessing Pipeline Utilities (V2.0)

Provides reusable helper functions for:
  - Output directory cleanup
  - Incoming dataset staging and copying
  - Dataset file metrics and counting
  - JSON manifest and pipeline report generation
  - Execution timing and console reporting formatting
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import global paths and parameters from config
try:
    from preprocessing.config import (
        AUTO_CLEAN_OUTPUT,
        AUTO_COPY_INCOMING,
        DATASET_DIR,
        DATASET_MANIFEST,
        INCOMING_DATASET_DIR,
        MANIFEST_DIR,
        METADATA_DIR,
        PIPELINE_REPORT,
        PIPELINE_VERSION,
        PROCESSED_DATASET_DIR,
        RAW_DATASET_DIR,
        REPORT_DIR,
        TEST_DIR,
        TRAIN_DIR,
        VAL_DIR,
    )
except ImportError:
    # Fallback default paths if running in isolated environment
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_DIR = PROJECT_ROOT / "datasets"
    INCOMING_DATASET_DIR = DATASET_DIR / "incoming"
    RAW_DATASET_DIR = DATASET_DIR / "raw"
    PROCESSED_DATASET_DIR = DATASET_DIR / "processed"
    METADATA_DIR = DATASET_DIR / "metadata"
    REPORT_DIR = DATASET_DIR / "reports"
    MANIFEST_DIR = DATASET_DIR / "manifest"
    TRAIN_DIR = DATASET_DIR / "train"
    VAL_DIR = DATASET_DIR / "val"
    TEST_DIR = DATASET_DIR / "test"
    DATASET_MANIFEST = MANIFEST_DIR / "dataset_manifest.json"
    PIPELINE_REPORT = REPORT_DIR / "pipeline_report.json"
    PIPELINE_VERSION = "2.0"
    AUTO_COPY_INCOMING = True
    AUTO_CLEAN_OUTPUT = True

logger = logging.getLogger("PixelSentinel.PipelineUtils")


# ========================================================== #
# 1. DIRECTORY CLEANUP & PREPARATION                        #
# ========================================================== #

def clean_directory_contents(target_dir: Path, extensions: Tuple[str, ...] = (".tif", ".png", ".jpg", ".json", ".csv")) -> int:
    """
    Recursively removes files matching given extensions in target_dir.
    Does not delete the directory itself.
    """
    if not target_dir.exists():
        return 0

    removed_count = 0
    for file_path in target_dir.rglob("*"):
        if file_path.is_file() and (not extensions or file_path.suffix.lower() in extensions):
            try:
                file_path.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not delete file {file_path}: {e}")

    return removed_count


def clean_processed_dataset() -> Dict[str, int]:
    """
    Cleans all intermediate processed outputs, split directories, and previous pipeline reports.
    Leaves raw/ and incoming/ intact.
    """
    logger.info("Cleaning previous dataset processing outputs...")
    
    deleted_counts = {

        "processed": clean_directory_contents(
            PROCESSED_DATASET_DIR,
        ),

        "train": clean_directory_contents(
            TRAIN_DIR,
        ),

        "val": clean_directory_contents(
            VAL_DIR,
        ),

        "test": clean_directory_contents(
            TEST_DIR,
        ),

        "reports": clean_directory_contents(
            REPORT_DIR,
        ),

        "manifest": clean_directory_contents(
            MANIFEST_DIR,
        ),

        "metadata": clean_directory_contents(
            METADATA_DIR,
            extensions=(".csv", ".json"),
        ),

    }

    total_deleted = sum(deleted_counts.values())
    logger.info(f"Cleanup complete. Total deleted items: {total_deleted}")
    return deleted_counts


# ========================================================== #
# 2. INCOMING DATASET MANAGEMENT                            #
# ========================================================== #

def copy_incoming_to_raw(overwrite: bool = True) -> Tuple[int, int]:
    """
    Copies all GeoTIFF files from the incoming directory
    to the raw dataset directory.
    Keeps the original files inside the incoming directory untouched as a backup.
    
    Returns:
        (copied_count, skipped_count)
    """
    INCOMING_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    incoming_files = sorted(
        list(INCOMING_DATASET_DIR.glob("*.tif"))
        +
        list(INCOMING_DATASET_DIR.glob("*.tiff"))
    )
    
    if not incoming_files:
        logger.info(f"No new GeoTIFF files found in {INCOMING_DATASET_DIR}. Proceeding with existing datasets/raw/.")
        return 0, 0

    logger.info(f"Found {len(incoming_files)} files in {INCOMING_DATASET_DIR}. Copying to {RAW_DATASET_DIR}...")
    
    copied = 0
    skipped = 0

    for file_path in incoming_files:
        dest_path = RAW_DATASET_DIR / file_path.name
        if dest_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            shutil.copy2(file_path, dest_path)
            copied += 1
        except Exception as e:
            logger.error(f"Failed to copy {file_path.name}: {e}")

    logger.info(f"Successfully copied {copied} scenes into {RAW_DATASET_DIR} (Skipped: {skipped}).")
    return copied, skipped


# ========================================================== #
# 3. METRICS & DATASET COUNTING                             #
# ========================================================== #

def count_files_by_extension(folder: Path, extensions: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg")) -> int:
    """Counts matching files recursively in a directory."""
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in extensions)


def get_dataset_counts() -> Dict[str, int]:
    """Collects current file counts across all dataset stages."""
    counts = {
        "incoming": count_files_by_extension(INCOMING_DATASET_DIR),
        "raw": count_files_by_extension(RAW_DATASET_DIR),
        "processed_input": count_files_by_extension(PROCESSED_DATASET_DIR / "input"),
        "processed_target": count_files_by_extension(PROCESSED_DATASET_DIR / "target"),
        "processed_tiles": count_files_by_extension(PROCESSED_DATASET_DIR / "tiles"),
        "processed_augmented": count_files_by_extension(PROCESSED_DATASET_DIR / "augmented"),
        "train": count_files_by_extension(TRAIN_DIR),
        "val": count_files_by_extension(VAL_DIR),
        "test": count_files_by_extension(TEST_DIR),
        "metadata": count_files_by_extension(
            METADATA_DIR, 
            extensions=(".csv", ".json"),
        ),
    }
    return counts


# ========================================================== #
# 4. MANIFEST & REPORT GENERATION                           #
# ========================================================== #

def write_dataset_manifest(
    stage_results: Dict[str, Any],
    total_execution_time: float,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generates datasets/manifest/dataset_manifest.json containing end-to-end dataset stats.
    Training scripts (train.py) can read this file directly to configure dataloaders.
    """
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    counts = get_dataset_counts()

    manifest_data = {
        "project": "PixelSentinel",
        "pipeline_version": PIPELINE_VERSION,
        "generated_by": "PixelSentinel Preprocessing Pipeline",
        "version": PIPELINE_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(total_execution_time, 2),
        "execution_time_formatted": format_time(total_execution_time),
        "dataset_counts": counts,
        "splits": {
            "train": counts["train"],
            "validation": counts["val"],
            "test": counts["test"]
        },
        "stage_execution_summary": stage_results
    }

    if additional_metadata:
        manifest_data["metadata"] = additional_metadata

    with open(DATASET_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    logger.info(f"Dataset manifest successfully written to: {DATASET_MANIFEST}")
    return DATASET_MANIFEST


def write_pipeline_report(report_data: Dict[str, Any]) -> Path:
    """Saves detailed logs and performance statistics to datasets/reports/pipeline_report.json."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PIPELINE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    logger.info(f"Pipeline execution report written to: {PIPELINE_REPORT}")
    return PIPELINE_REPORT


# ========================================================== #
# 5. TIMER & FORMATTING UTILITIES                           #
# ========================================================== #

class PipelineTimer:
    """Context manager and utility to track execution duration."""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self) -> float:
        self.end_time = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        current = self.end_time if self.end_time else time.time()
        return current - self.start_time


def format_time(seconds: float) -> str:
    """Formats time in seconds into human-readable string (e.g., '1h 12m 30s')."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{seconds:.2f}s"


def print_pipeline_summary(stage_results: Dict[str, Any], total_time: float) -> None:
    """Prints a clean CLI summary table of all executed pipeline stages."""
    counts = get_dataset_counts()
    
    print("\n" + "=" * 70)
    print(f"         PIXELSENTINEL PREPROCESSING PIPELINE V{PIPELINE_VERSION} SUMMARY         ")
    print("=" * 70)
    print(f" Total Execution Time : {format_time(total_time)}")
    print("-" * 70)
    print(f" {'Stage':<25} | {'Status':<10} | {'Processed':<12} | {'Time':<10}")
    print("-" * 70)

    for stage_name, res in stage_results.items():
        status = "PASSED" if res.get("success", False) else "FAILED"
        processed = str(res.get("processed", 0))
        exec_time = format_time(res.get("execution_time", 0.0))
        print(f" {stage_name:<25} | {status:<10} | {processed:<12} | {exec_time:<10}")

    print("-" * 70)
    print(" DATASET SPLIT FINAL COUNTS:")
    print(f"  • Raw Input Scenes : {counts['raw']}")
    print(f"  • Train Set Tiles  : {counts['train']}")
    print(f"  • Val Set Tiles    : {counts['val']}")
    print(f"  • Test Set Tiles   : {counts['test']}")
    print("=" * 70 + "\n")

def pipeline_header(title: str):

    logger.info("=" * 80)

    logger.info(title)

    logger.info("=" * 80)

def pipeline_footer():
    logger.info("=" * 80)

def dataset_exists() -> bool:

    return any(
        RAW_DATASET_DIR.glob("*.tif")
    )

if __name__ == "__main__":

    if not dataset_exists():

        raise RuntimeError(
            "No raw input scenes found."
        )

    pipeline_footer()