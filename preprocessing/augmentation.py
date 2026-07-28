"""
PixelSentinel Dataset Augmentation

Offline paired augmentation for Pix2Pix training.

Applies identical geometric transformations to
input and target tiles.

Operations
----------
✓ Horizontal Flip
✓ Vertical Flip
✓ Rotate 90°
✓ Rotate 180°
✓ Rotate 270°

Author
------
PixelSentinel Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import rasterio

from preprocessing.config import (
    INPUT_TILE_DIR,
    TARGET_TILE_DIR,
    PROCESSED_DATASET_DIR,
)

from preprocessing.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# Output Directories
# ==========================================================

AUGMENTED_DIR = (
    PROCESSED_DATASET_DIR /
    "augmented"
)

INPUT_OUTPUT_DIR = (
    AUGMENTED_DIR /
    "input"
)

TARGET_OUTPUT_DIR = (
    AUGMENTED_DIR /
    "target"
)

INPUT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ==========================================================
# Result
# ==========================================================

@dataclass(slots=True)
class AugmentationResult:

    filename: str

    success: bool

    augmentations_created: int

    message: str


# ==========================================================
# Dataset Augmentor
# ==========================================================

class DatasetAugmentor:

    """
    Offline paired augmentation.
    """

    def __init__(self) -> None:

        self.input_directory = INPUT_TILE_DIR

        self.target_directory = TARGET_TILE_DIR

        self.output_input_directory = INPUT_OUTPUT_DIR

        self.output_target_directory = TARGET_OUTPUT_DIR

        self.transforms = [

            (
                "hflip",
                A.HorizontalFlip(
                    p=1.0,
                ),
            ),

            (
                "vflip",
                A.VerticalFlip(
                    p=1.0,
                ),
            ),

            (
                "rot90",
                A.RandomRotate90(
                    p=1.0,
                ),
            ),

            (
                "rot180",
                A.Rotate(
                    limit=(180, 180),
                    crop_border=False,
                    p=1.0,
                ),
            ),

            (
                "rot270",
                A.Rotate(
                    limit=(270, 270),
                    crop_border=False,
                    p=1.0,
                ),
            ),

        ]

    # ======================================================
    # Helpers
    # ======================================================

    @staticmethod
    def paired_target(
        input_tile: Path,
    ) -> Path:

        return (
            TARGET_TILE_DIR /
            input_tile.name.replace(
                "_input.tif",
                "_target.tif",
            )
        )

    @staticmethod
    def create_profile(
        dataset: rasterio.io.DatasetReader,
        band_count: int,
    ) -> dict[str, Any]:

        profile = dataset.profile.copy()

        profile.update(
            count=band_count,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        return profile

    @staticmethod
    def save_geotiff(
        output_path: Path,
        image: np.ndarray,
        profile: dict[str, Any],
        descriptions: tuple[str, ...],
    ) -> None:

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as destination:

            destination.write(image)

            destination.descriptions = descriptions

    # ======================================================
    # Apply Transform
    # ======================================================

    @staticmethod
    def apply_transform(
        transform,
        input_image: np.ndarray,
        target_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        input_hwc = np.transpose(
            input_image,
            (1, 2, 0),
        )

        target_hwc = np.transpose(
            target_image,
            (1, 2, 0),
        )

        augmented = transform(
            image=input_hwc,
            mask=target_hwc,
        )

        input_result = np.transpose(
            augmented["image"],
            (2, 0, 1),
        )

        target_result = np.transpose(
            augmented["mask"],
            (2, 0, 1),
        )

        return (
            input_result,
            target_result,
        )


    # ======================================================
    # Process One Tile Pair
    # ======================================================

    def process_tile(
        self,
        input_tile: Path,
    ) -> AugmentationResult:

        target_tile = self.paired_target(
            input_tile,
        )

        if not target_tile.exists():

            return AugmentationResult(
                filename=input_tile.name,
                success=False,
                augmentations_created=0,
                message="Missing target tile.",
            )

        created = 0

        try:

            with rasterio.open(input_tile) as input_ds, \
                 rasterio.open(target_tile) as target_ds:

                input_image = input_ds.read()

                target_image = target_ds.read()

                input_profile = self.create_profile(
                    input_ds,
                    input_ds.count,
                )

                target_profile = self.create_profile(
                    target_ds,
                    target_ds.count,
                )

                input_desc = tuple(
                    input_ds.descriptions
                )

                target_desc = tuple(
                    target_ds.descriptions
                )
                # ------------------------------------------
                # Save Original Tile
                # ------------------------------------------

                original_input = (
                    self.output_input_directory /
                    input_tile.name
                )

                original_target = (
                    self.output_target_directory /
                    target_tile.name
                )

                self.save_geotiff(
                    original_input,
                    input_image,
                    input_profile,
                    input_desc,
                )

                self.save_geotiff(
                    original_target,
                    
                    target_image,
                    target_profile,
                    target_desc,
                )

                created += 1

                # ------------------------------------------
                # Generate Augmentations
                # ------------------------------------------

                for suffix, transform in self.transforms:

                    augmented_input, augmented_target = (
                        self.apply_transform(
                            transform,
                            input_image,
                            target_image,
                        )
                    )

                    input_name = (
                        input_tile.stem
                        + f"_{suffix}.tif"
                    )

                    target_name = (
                        target_tile.stem
                        + f"_{suffix}.tif"
                    )

                    self.save_geotiff(
                        self.output_input_directory / input_name,
                        augmented_input,
                        input_profile,
                        input_desc,
                    )

                    self.save_geotiff(
                        self.output_target_directory / target_name,
                        augmented_target,
                        target_profile,
                        target_desc,
                    )

                    created += 1

            return AugmentationResult(
                filename=input_tile.name,
                success=True,
                augmentations_created=created,
                message="Success",
            )

        except Exception as exc:

            logger.exception(
                "Failed processing %s",
                input_tile.name,
            )

            return AugmentationResult(
                filename=input_tile.name,
                success=False,
                augmentations_created=0,
                message=str(exc),
            )

    # ======================================================
    # Process Dataset
    # ======================================================

    def process_dataset(
        self,
    ) -> None:
        """
        Augment the complete tiled dataset.
        """

        start_time = time.time()

        logger.info("=" * 80)
        logger.info("STARTING DATASET AUGMENTATION")
        logger.info("=" * 80)

        input_tiles = sorted(
            self.input_directory.glob("*_input.tif")
        )

        logger.info(
            "Found %d input tiles.",
            len(input_tiles),
        )

        successful = 0
        failed = 0
        total_augmented = 0

        for index, input_tile in enumerate(
            input_tiles,
            start=1,
        ):

            result = self.process_tile(
                input_tile,
            )

            if result.success:

                successful += 1
                total_augmented += (
                    result.augmentations_created
                )

            else:

                failed += 1

            if (
                index % 500 == 0
                or index == len(input_tiles)
            ):

                logger.info(
                    "Processed %d / %d tiles...",
                    index,
                    len(input_tiles),
                )

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("AUGMENTATION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            "Input Tiles          : %d",
            len(input_tiles),
        )
        logger.info(
            "Successful           : %d",
            successful,
        )
        logger.info(
            "Failed               : %d",
            failed,
        )
        logger.info(
            "Augmented Tile Pairs : %d",
            total_augmented,
        )
        logger.info("=" * 80)
        logger.info(
            "Execution Time : %.2f seconds",
            elapsed,
        )
        logger.info("=" * 80)

        if failed == 0:

            logger.info(
                "Dataset augmentation completed successfully."
            )

        else:

            logger.warning(
                "%d tiles failed during augmentation.",
                failed,
            )

# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Execute offline augmentation.
    """

    augmentor = DatasetAugmentor()

    augmentor.process_dataset()


if __name__ == "__main__":

    main()