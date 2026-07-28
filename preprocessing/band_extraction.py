"""
PixelSentinel Band Extraction Module

Extracts the Pix2Pix input and target bands from the raw Landsat
GeoTIFF scenes while preserving all geospatial metadata.

Input Bands
-----------
SR_B5
SR_B6
SR_B7
ST_B10

Target Bands
------------
SR_B4
SR_B3
SR_B2

Outputs
-------
datasets/processed/

    input/
        *_input.tif

    target/
        *_target.tif
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio
from rasterio.errors import RasterioIOError
import time
from preprocessing.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    INPUT_BANDS,
    TARGET_BANDS,
    EXPECTED_BAND_ORDER,
    SUPPORTED_FILE_EXTENSIONS,
)

from preprocessing.logger import get_logger
from preprocessing.resource_monitor import (
    monitor_progress,
)

logger = get_logger(__name__)


# ==========================================================
# Output Directories
# ==========================================================

INPUT_OUTPUT_DIR = PROCESSED_DATASET_DIR / "input"
TARGET_OUTPUT_DIR = PROCESSED_DATASET_DIR / "target"

INPUT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Dataclass
# ==========================================================

@dataclass(slots=True)
class BandExtractionResult:
    """
    Result of processing one GeoTIFF scene.
    """

    filename: str

    success: bool

    input_path: Path | None

    target_path: Path | None

    message: str


# ==========================================================
# Band Extractor
# ==========================================================

class BandExtractor:
    """
    Extract input and target bands from Landsat GeoTIFF scenes.

    The extractor preserves all important geospatial metadata,
    including CRS, affine transform, resolution, NoData values,
    compression, and driver information.
    """

    def __init__(
        self,
        raw_directory: Path = RAW_DATASET_DIR,
        input_directory: Path = INPUT_OUTPUT_DIR,
        target_directory: Path = TARGET_OUTPUT_DIR,
    ) -> None:

        self.raw_directory = raw_directory

        self.input_directory = input_directory

        self.target_directory = target_directory

        logger.info("BandExtractor initialized.")

        logger.info("Raw Dataset      : %s", raw_directory)

        logger.info("Input Output     : %s", input_directory)

        logger.info("Target Output    : %s", target_directory)

    # ======================================================
    # Validation Helpers
    # ======================================================

    @staticmethod
    def _supported_file(path: Path) -> bool:
        """
        Check supported file extension.
        """

        return path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS

    @staticmethod
    def _validate_band_count(
        dataset: rasterio.DatasetReader,
    ) -> bool:
        """
        Ensure the scene contains exactly seven bands.
        """

        return dataset.count == len(EXPECTED_BAND_ORDER)

    @staticmethod
    def _validate_band_order(
        dataset: rasterio.DatasetReader,
    ) -> bool:
        """
        Validate exported Landsat band descriptions.
        """

        descriptions = [
            item.strip() if item else ""
            for item in dataset.descriptions
        ]

        return descriptions == EXPECTED_BAND_ORDER

    def validate_scene(
        self,
        dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Validate a scene before extraction.
        """

        if not self._validate_band_count(dataset):

            raise ValueError(
                f"Expected {len(EXPECTED_BAND_ORDER)} bands "
                f"but found {dataset.count}."
            )

        if not self._validate_band_order(dataset):

            raise ValueError(
                "Band descriptions do not match "
                "PixelSentinel configuration."
            )

    # ======================================================
    # Metadata
    # ======================================================

    @staticmethod
    def _create_profile(
        dataset: rasterio.DatasetReader,
        band_count: int,
    ) -> dict[str, Any]:
        """
        Create output GeoTIFF profile while preserving
        spatial metadata.
        """

        profile = dataset.profile.copy()

        profile.update(
            count=band_count
        )

        return profile

    @staticmethod
    def _output_paths(
        scene_path: Path,
        input_directory: Path,
        target_directory: Path,
    ) -> tuple[Path, Path]:
        """
        Generate output filenames.
        """

        stem = scene_path.stem

        input_path = (
            input_directory /
            f"{stem}_input.tif"
        )

        target_path = (
            target_directory /
            f"{stem}_target.tif"
        )

        return input_path, target_path

    # ======================================================
    # Scene Opening
    # ======================================================

    @staticmethod
    def open_scene(
        scene_path: Path,
    ) -> rasterio.DatasetReader:
        """
        Open a GeoTIFF safely.
        """

        if not scene_path.exists():

            raise FileNotFoundError(scene_path)

        try:

            return rasterio.open(scene_path)

        except RasterioIOError as exc:

            logger.exception(
                "Unable to open %s",
                scene_path,
            )

            raise exc

    # ======================================================
    # Band Extraction
    # ======================================================

    @staticmethod
    def extract_input(
        dataset: rasterio.DatasetReader,
    ):
        """
        Extract Pix2Pix input bands.

        Returns
        -------
        ndarray
            Shape:
            (4, H, W)
        """

        input_indexes = [
            EXPECTED_BAND_ORDER.index(band) + 1
            for band in INPUT_BANDS
        ]

        logger.info(
            "Extracting input bands: %s",
            ", ".join(INPUT_BANDS),
        )

        return dataset.read(input_indexes)

    @staticmethod
    def extract_target(
        dataset: rasterio.DatasetReader,
    ):
        """
        Extract Pix2Pix RGB target bands.

        Returns
        -------
        ndarray
            Shape:
            (3, H, W)
        """

        target_indexes = [
            EXPECTED_BAND_ORDER.index(band) + 1
            for band in TARGET_BANDS
        ]

        logger.info(
            "Extracting target bands: %s",
            ", ".join(TARGET_BANDS),
        )

        return dataset.read(target_indexes)

    # ======================================================
    # Saving
    # ======================================================

    @staticmethod
    def save_geotiff(
        output_path: Path,
        image,
        profile: dict[str, Any],
        descriptions: list[str],
    ) -> None:
        """
        Save GeoTIFF while preserving metadata.
        """

        profile = profile.copy()

        profile.update(
            count=image.shape[0],
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as destination:

            destination.write(image)
            destination.descriptions = tuple(descriptions)

        logger.info(
            "Saved %s",
            output_path.name,
        )

    # ======================================================
    # Scene Processing
    # ======================================================

    def process_scene(
        self,
        scene_path: Path,
    ) -> BandExtractionResult:
        """
        Process a single GeoTIFF scene.
        """

        logger.info(
            "-" * 70,
        )

        logger.info(
            "Processing %s",
            scene_path.name,
        )

        input_path, target_path = self._output_paths(
            scene_path,
            self.input_directory,
            self.target_directory,
        )

        if input_path.exists() and target_path.exists():

            logger.info(
                "Skipping existing scene %s",
                scene_path.name,
            )

            return BandExtractionResult(
                filename=scene_path.name,
                success=True,
                input_path=input_path,
                target_path=target_path,
                message="Already processed",
            )

        try:

            with self.open_scene(scene_path) as dataset:

                self.validate_scene(dataset)

                input_image = self.extract_input(
                    dataset,
                )

                target_image = self.extract_target(
                    dataset,
                )

                input_profile = self._create_profile(
                    dataset,
                    len(INPUT_BANDS),
                )

                target_profile = self._create_profile(
                    dataset,
                    len(TARGET_BANDS),
                )

                self.save_geotiff(
                    input_path,
                    input_image,
                    input_profile,
                    INPUT_BANDS,
                )
 
                self.save_geotiff(
                    target_path,
                    target_image,
                    target_profile,
                    TARGET_BANDS,
                )

            logger.info(
                "Completed %s",
                scene_path.name,
            )

            return BandExtractionResult(
                filename=scene_path.name,
                success=True,
                input_path=input_path,
                target_path=target_path,
                message="Success",
            )

        except Exception as exc:

            logger.exception(
                "Failed processing %s",
                scene_path.name,
            )

            return BandExtractionResult(
                filename=scene_path.name,
                success=False,
                input_path=None,
                target_path=None,
                message=str(exc),
            )


    # ======================================================
    # Dataset Processing
    # ======================================================

    def process_dataset(
        self,
    ) -> list[BandExtractionResult]:
        """
        Process every GeoTIFF inside the raw dataset directory.

        Returns
        -------
        list[BandExtractionResult]
        """

        scene_paths = sorted(
            [
                path
                for path in self.raw_directory.iterdir()
                if path.is_file()
                and self._supported_file(path)
            ]
        )

        logger.info("=" * 80)
        logger.info("Starting Band Extraction")
        logger.info("=" * 80)

        logger.info(
            "Found %d GeoTIFF scenes.",
            len(scene_paths),
        )

        results: list[BandExtractionResult] = []

        total_files = len(scene_paths)

        for index, scene_path in enumerate(scene_paths, start=1):

            monitor_progress(
                current=index,
                total=total_files,
                every=100,
            )
            result = self.process_scene(scene_path)

            results.append(result)
        return results


# ==========================================================
# Summary
# ==========================================================


def print_summary(
    results: list[BandExtractionResult],
) -> None:
    """
    Print extraction summary.
    """

    total = len(results)

    success = sum(
        result.success
        for result in results
    )

    failed = total - success

    logger.info("=" * 80)
    logger.info("Band Extraction Summary")
    logger.info("=" * 80)

    logger.info(
        "Scenes Processed : %d",
        total,
    )

    logger.info(
        "Successful       : %d",
        success,
    )

    logger.info(
        "Failed           : %d",
        failed,
    )

    if failed:

        logger.info("-" * 80)

        logger.info("Failed Files")

        logger.info("-" * 80)

        for result in results:

            if not result.success:

                logger.info(
                    "%s -> %s",
                    result.filename,
                    result.message,
                )

    logger.info("=" * 80)


def extract_bands():
    """
    Pipeline entry point.
    Returns a dictionary understood by PipelineManager.
    """

    start_time = time.perf_counter()

    extractor = BandExtractor()

    results = extractor.process_dataset()

    elapsed = time.perf_counter() - start_time

    success = sum(r.success for r in results)

    failed = len(results) - success

    print_summary(results)

    logger.info(
        "Execution Time : %.2f seconds",
        elapsed,
    )

    return {
        "success": failed == 0,
        "processed": success,
        "failed": failed,
        "execution_time": elapsed,
    }

# ==========================================================
# Main
# ==========================================================


def main() -> None:
    """
    Execute the complete band extraction pipeline.
    """

    start_time = time.perf_counter()

    extractor = BandExtractor()

    results = extractor.process_dataset()

    elapsed = time.perf_counter() - start_time

    print_summary(results)

    logger.info(
        "Execution Time : %.2f seconds",
        elapsed,
    )

def process_dataset():
    """
    Entry point for the Pipeline Manager to execute band extraction.
    """
    # Assuming your main class is named BandExtractor
    extractor = BandExtractor() 
    results = extractor.process_dataset()
    
    # Calculate passes and failures
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    
    return {
        "success": failed == 0,
        "processed": passed,
        "failed": failed,
    }

if __name__ == "__main__":

    main()