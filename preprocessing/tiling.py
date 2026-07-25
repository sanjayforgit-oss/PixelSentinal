"""
PixelSentinel Image Tiling Module

Splits normalized paired GeoTIFFs into fixed-size patches while
preserving geospatial metadata.

Features
--------
- Boundary-aware tiling
- Input/Target synchronized tiles
- Preserves CRS and affine transform
- Tile metadata generation
- Progress logging
"""

from __future__ import annotations

import csv
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from preprocessing.config import (
    PROCESSED_DATASET_DIR,
    METADATA_DIR,
    TILE_SIZE,
    TILE_STRIDE,
    USE_BOUNDARY_OVERLAP,
    GENERATE_TILE_METADATA,
    TILE_COMPRESSION,
    INPUT_TILE_DIR,
    TARGET_TILE_DIR,
    TILE_METADATA_FILE,
)

from preprocessing.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Normalized Dataset Directories
# ==========================================================

NORMALIZED_INPUT_DIR = (
    PROCESSED_DATASET_DIR
    / "normalized"
    / "input"
)

NORMALIZED_TARGET_DIR = (
    PROCESSED_DATASET_DIR
    / "normalized"
    / "target"
)


# ==========================================================
# Tile Result
# ==========================================================

@dataclass(slots=True)
class TileResult:
    """
    Result of processing one scene pair.
    """

    filename: str

    success: bool

    tiles_created: int

    message: str


# ==========================================================
# Image Tiler
# ==========================================================

class ImageTiler:
    """
    Production-grade GeoTIFF tiler.

    Generates perfectly aligned paired tiles from
    normalized input and target imagery.
    """

    def __init__(self) -> None:

        self.tile_size = TILE_SIZE
        self.stride = TILE_STRIDE

        self.input_directory = NORMALIZED_INPUT_DIR
        self.target_directory = NORMALIZED_TARGET_DIR

        self.output_input_directory = INPUT_TILE_DIR
        self.output_target_directory = TARGET_TILE_DIR

        self.metadata_file = TILE_METADATA_FILE

        self.output_input_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_target_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if GENERATE_TILE_METADATA:

            if self.metadata_file.exists():
                self.metadata_file.unlink()

            self._initialize_metadata()


    # ======================================================
    # Metadata
    # ======================================================

    def _initialize_metadata(self) -> None:
        """
        Create tile metadata CSV.
        """

        header = [

            "Scene",

            "Tile_ID",

            "Row",

            "Column",

            "Pixel_X",

            "Pixel_Y",

            "Input_File",

            "Target_File",

        ]

        with open(
            self.metadata_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(header)


    def append_metadata(
        self,
        row: list[Any],
    ) -> None:
        """
        Append one metadata record.
        """

        with open(
            self.metadata_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(row)


    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def scene_name(path: Path) -> str:
        """
        Remove suffix from filename.
        """

        name = path.stem

        name = name.replace("_input", "")

        name = name.replace("_target", "")

        return name


    @staticmethod
    def paired_target(
        input_file: Path,
    ) -> Path:
        """
        Return corresponding target filename.
        """

        return (
            NORMALIZED_TARGET_DIR
            /
            input_file.name.replace(
                "_input",
                "_target",
            )
        )


    @staticmethod
    def create_profile(
        dataset: rasterio.DatasetReader,
        transform,
        bands: int,
    ) -> dict[str, Any]:
        """
        Create output GeoTIFF profile.
        """

        profile = dataset.profile.copy()

        profile.update(

            width=TILE_SIZE,

            height=TILE_SIZE,

            transform=transform,

            count=bands,

            compress=TILE_COMPRESSION,

            tiled=True,

            BIGTIFF="IF_SAFER",

        )

        return profile


    # ======================================================
    # Tile Position Generation
    # ======================================================

    def generate_positions(
        self,
        image_size: int,
    ) -> list[int]:
        """
        Generate tile starting positions.

        Uses boundary-aware overlap so that the final tile always
        reaches the image boundary without padding or cropping.

        Example
        -------
        Image width = 1637
        Tile size   = 256

        Positions:
        0, 256, 512, 768, 1024, 1280, 1381

        The final tile starts at 1381 so that:

            1381 + 256 = 1637

        Parameters
        ----------
        image_size : int
            Width or height of the image.

        Returns
        -------
        list[int]
            Tile starting coordinates.
        """

        if image_size <= self.tile_size:
            return [0]

        positions: list[int] = []

        current = 0

        while current + self.tile_size <= image_size:

            positions.append(current)

            current += self.stride

        if USE_BOUNDARY_OVERLAP:

            last_position = image_size - self.tile_size

            if positions[-1] != last_position:

                positions.append(last_position)

        return sorted(set(positions))


    # ======================================================
    # Tile Window Generator
    # ======================================================

    def generate_windows(
        self,
        width: int,
        height: int,
    ) -> list[tuple[int, int, Window]]:
        """
        Generate rasterio windows covering the entire image.

        Returns
        -------
        list
            (row_index,
             column_index,
             rasterio.Window)
        """

        x_positions = self.generate_positions(width)

        y_positions = self.generate_positions(height)

        windows: list[tuple[int, int, Window]] = []

        for row_index, y in enumerate(y_positions):

            for column_index, x in enumerate(x_positions):

                windows.append(

                    (
                        row_index,
                        column_index,
                        Window(
                            col_off=x,
                            row_off=y,
                            width=self.tile_size,
                            height=self.tile_size,
                        ),
                    )

                )

        return windows


    # ======================================================
    # Save Tile
    # ======================================================

    def save_tile(
        self,
        output_path: Path,
        tile: np.ndarray,
        profile: dict[str, Any],
        descriptions: tuple[str, ...],
    ) -> None:
        """
        Save a single GeoTIFF tile.
        """

        profile = profile.copy()

        profile.update(
            count=tile.shape[0],
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as destination:

            destination.write(tile)

            destination.descriptions = descriptions


    # ======================================================
    # Tile Name
    # ======================================================

    @staticmethod
    def tile_name(
        scene: str,
        tile_id: int,
        suffix: str,
    ) -> str:
        """
        Build deterministic tile filename.

        Example
        -------
        Landsat9_Delhi_2025_Summer_001_tile_000123_input.tif
        """

        return (
            f"{scene}"
            f"_tile_{tile_id:06d}"
            f"_{suffix}.tif"
        )

    # ======================================================
    # Process Scene Pair
    # ======================================================

    def process_scene(
        self,
        input_path: Path,
    ) -> TileResult:
        """
        Generate paired tiles for one normalized scene.
        """

        scene = self.scene_name(input_path)

        target_path = self.paired_target(input_path)

        if not target_path.exists():

            return TileResult(
                filename=scene,
                success=False,
                tiles_created=0,
                message="Matching target not found.",
            )

        logger.info("Processing %s", scene)

        tile_counter = 0

        try:

            with rasterio.open(input_path) as input_ds, \
                 rasterio.open(target_path) as target_ds:

                windows = self.generate_windows(
                    input_ds.width,
                    input_ds.height,
                )

                input_descriptions = tuple(
                    input_ds.descriptions
                )

                target_descriptions = tuple(
                    target_ds.descriptions
                )

                for row_index, column_index, window in windows:

                    input_tile = input_ds.read(
                        window=window,
                    )

                    target_tile = target_ds.read(
                        window=window,
                    )

                    transform = window_transform(
                        window,
                        input_ds.transform,
                    )

                    input_profile = self.create_profile(
                        input_ds,
                        transform,
                        input_tile.shape[0],
                    )

                    target_profile = self.create_profile(
                        target_ds,
                        transform,
                        target_tile.shape[0],
                    )

                    tile_counter += 1

                    input_filename = self.tile_name(
                        scene,
                        tile_counter,
                        "input",
                    )

                    target_filename = self.tile_name(
                        scene,
                        tile_counter,
                        "target",
                    )

                    input_output = (
                        self.output_input_directory
                        / input_filename
                    )

                    target_output = (
                        self.output_target_directory
                        / target_filename
                    )

                    self.save_tile(
                        input_output,
                        input_tile,
                        input_profile,
                        input_descriptions,
                    )

                    self.save_tile(
                        target_output,
                        target_tile,
                        target_profile,
                        target_descriptions,
                    )

                    if GENERATE_TILE_METADATA:

                        self.append_metadata(

                            [

                                scene,

                                tile_counter,

                                row_index,

                                column_index,

                                int(window.col_off),

                                int(window.row_off),

                                input_filename,

                                target_filename,

                            ]

                        )

            logger.info(
                "Created %d tiles for %s",
                tile_counter,
                scene,
            )

            return TileResult(
                filename=scene,
                success=True,
                tiles_created=tile_counter,
                message="Success",
            )

        except Exception as exc:

            logger.exception(
                "Failed processing %s",
                scene,
            )

            return TileResult(
                filename=scene,
                success=False,
                tiles_created=0,
                message=str(exc),
            )

    # ======================================================
    # Process Dataset
    # ======================================================

    def process_dataset(self) -> None:
        """
        Process the complete normalized dataset.
        """

        start_time = time.time()

        logger.info("=" * 80)
        logger.info("STARTING IMAGE TILING")
        logger.info("=" * 80)

        input_files = sorted(
            self.input_directory.glob("*_input.tif")
        )

        logger.info(
            "Found %d normalized input scenes.",
            len(input_files),
        )

        successful = 0
        failed = 0
        total_tiles = 0

        for index, input_file in enumerate(input_files, start=1):

            result = self.process_scene(
                input_file,
            )

            if result.success:

                successful += 1
                total_tiles += result.tiles_created

            else:

                failed += 1

            if (
                index % 25 == 0
                or index == len(input_files)
            ):

                logger.info(
                    "Processed %d / %d scenes...",
                    index,
                    len(input_files),
                )

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("IMAGE TILING SUMMARY")
        logger.info("=" * 80)
        logger.info(
            "Scenes Processed : %d",
            len(input_files),
        )
        logger.info(
            "Successful      : %d",
            successful,
        )
        logger.info(
            "Failed          : %d",
            failed,
        )
        logger.info(
            "Tiles Generated : %d",
            total_tiles,
        )
        logger.info("=" * 80)
        logger.info(
            "Execution Time : %.2f seconds",
            elapsed,
        )
        logger.info("=" * 80)

        if failed == 0:

            logger.info(
                "Dataset tiling completed successfully."
            )

        else:

            logger.warning(
                "%d scene(s) failed during tiling.",
                failed,
            )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Execute the image tiling pipeline.
    """

    tiler = ImageTiler()

    tiler.process_dataset()


if __name__ == "__main__":

    main()