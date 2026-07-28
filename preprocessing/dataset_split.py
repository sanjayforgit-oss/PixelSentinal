"""
PixelSentinel — Dataset Splitter Module (V2.0)

Splits paired tile datasets (input_tiles and target_tiles) into Train, Validation, 
and Test directories according to config ratios, ensuring reproducible random shuffling.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from preprocessing.resource_monitor import (
    monitor_progress,
)
try:
    from preprocessing.config import (
        DATASET_DIR,
        INPUT_DIR_NAME,
        PROCESSED_DATASET_DIR,
        TARGET_DIR_NAME,
        TEST_DIR,
        TEST_RATIO,
        TRAIN_DIR,
        TRAIN_RATIO,
        VAL_DIR,
        VALIDATION_RATIO,
    )
    from preprocessing.pipeline_manager import StageResult
except ImportError:
    # Fallback paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_DIR = PROJECT_ROOT / "datasets"
    PROCESSED_DATASET_DIR = DATASET_DIR / "processed"
    INPUT_DIR_NAME = "input"
    TARGET_DIR_NAME = "target"
    TRAIN_DIR = DATASET_DIR / "train"
    VAL_DIR = DATASET_DIR / "val"
    TEST_DIR = DATASET_DIR / "test"
    TRAIN_RATIO = 0.80
    VALIDATION_RATIO = 0.10
    TEST_RATIO = 0.10

    class StageResult:
        def __init__(self, stage_name, success, processed, failed, execution_time=0.0, details=None):
            self.stage_name = stage_name
            self.success = success
            self.processed = processed
            self.failed = failed
            self.execution_time = execution_time
            self.details = details or {}

logger = logging.getLogger("PixelSentinel.DatasetSplitter")


def setup_split_directories():
    """Creates destination train, val, and test subdirectories."""
    for root_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        if root_dir.exists():

            shutil.rmtree(root_dir)

        (root_dir / INPUT_DIR_NAME).mkdir(
            parents=True,
            exist_ok=True,
        )

        (root_dir / TARGET_DIR_NAME).mkdir(
            parents=True,
            exist_ok=True,
        )


def get_paired_tiles() -> List[Tuple[Path, Path]]:
    """Retrieves matched pairs of input and target tiles using base filenames."""
    
    from preprocessing.config import (
        INPUT_TILE_DIR,
        TARGET_TILE_DIR,
    )

    candidate_paths = [
        (
            INPUT_TILE_DIR,
            TARGET_TILE_DIR,
        )
    ]

    input_dir = None
    target_dir = None

    for in_path, tgt_path in candidate_paths:
        if in_path.exists() and tgt_path.exists():
            input_dir = in_path
            target_dir = tgt_path
            break

    if not input_dir or not target_dir:
        logger.error(f"Tile directories missing in {PROCESSED_DATASET_DIR}")
        return []

    logger.info(f"Loading paired tiles from:\n  • Input : {input_dir}\n  • Target: {target_dir}")

    # Create dictionaries using the BASE NAME (stripping _input.tif and _target.tif)
    input_files = {f.name.replace("_input.tif", ""): f for f in input_dir.glob("*.tif")}
    target_files = {f.name.replace("_target.tif", ""): f for f in target_dir.glob("*.tif")}

    # Find common base names
    common_names = sorted(list(set(input_files.keys()) & set(target_files.keys())))
    pairs = [(input_files[name], target_files[name]) for name in common_names]

    logger.info(f"Found {len(pairs)} matched tile pairs for dataset splitting.")
    return pairs

def copy_pair_list(
    pairs: List[Tuple[Path, Path]],
    dest_dir: Path,
) -> int:
    """
    Copies paired input/target tiles into the destination split.

    Existing files are skipped to support resuming an interrupted
    pipeline execution.
    """

    dest_input = dest_dir / INPUT_DIR_NAME
    dest_target = dest_dir / TARGET_DIR_NAME

    copied = 0

    total = len(pairs)

    for index, (input_tile, target_tile) in enumerate(
        pairs,
        start=1,
    ):

        monitor_progress(
            current=index,
            total=total,
            every=250,
        )

        try:

            input_destination = (
                dest_input /
                input_tile.name
            )

            target_destination = (
                dest_target /
                target_tile.name
            )

            #
            # Skip if both already exist
            #

            if (
                input_destination.exists()
                and
                target_destination.exists()
            ):
                copied += 1
                continue

            #
            # Copy input
            #

            if not input_destination.exists():

                shutil.copy2(
                    input_tile,
                    input_destination,
                )

            #
            # Copy target
            #

            if not target_destination.exists():

                shutil.copy2(
                    target_tile,
                    target_destination,
                )

            copied += 1

        except Exception as exc:

            logger.exception(
                "Failed copying tile pair %s",
                input_tile.name,
            )

    logger.info(
        "Copied %d / %d tile pairs to %s",
        copied,
        total,
        dest_dir.name,
    )

    return copied


def split_dataset(seed: int = 42) -> StageResult:
    """Main entry point for splitting the dataset into train/val/test splits."""
    logger.info("================================================================================")
    logger.info("STARTING DATASET PARTITIONING (TRAIN / VAL / TEST)")
    logger.info("================================================================================")

    pairs = get_paired_tiles()

    if not pairs:
        logger.error("No valid tile pairs found to split!")
        return StageResult(
            stage_name="Dataset Split",
            success=False,
            processed=0,
            failed=0,
            details={"error": "No tile pairs found in tiles directory"}
        )

    random.seed(seed)
    random.shuffle(pairs)

    total_pairs = len(pairs)
    train_count = int(total_pairs * TRAIN_RATIO)
    val_count = int(total_pairs * VALIDATION_RATIO)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:train_count + val_count]
    test_pairs = pairs[train_count + val_count:]

    setup_split_directories()

    logger.info(f"Partitioning {total_pairs} pairs -> Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    copied_train = copy_pair_list(train_pairs, TRAIN_DIR)
    copied_val = copy_pair_list(val_pairs, VAL_DIR)
    copied_test = copy_pair_list(test_pairs, TEST_DIR)

    total_copied = copied_train + copied_val + copied_test
    success = (total_copied == total_pairs)

    logger.info("Dataset split complete:")
    logger.info(f"  • Train : {copied_train} pairs -> {TRAIN_DIR}")
    logger.info(f"  • Val   : {copied_val} pairs -> {VAL_DIR}")
    logger.info(f"  • Test  : {copied_test} pairs -> {TEST_DIR}")

    return StageResult(
        stage_name="Dataset Split",
        success=success,
        processed=total_copied,
        failed=total_pairs - total_copied,
        details={
            "train": copied_train,
            "validation": copied_val,
            "test": copied_test
        }
    )

def process_dataset():
    """
    Entry point for the Pipeline Manager to execute the dataset split.
    """
    # Run the split logic
    result = split_dataset()
    
    # Extract the values from the StageResult object into the expected dictionary
    return {
        "success": result.success,
        "processed": result.processed,
        "failed": result.failed,
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    result = split_dataset()
    print(f"\nExecution Finished. Status: {'SUCCESS' if result.success else 'FAILED'}")