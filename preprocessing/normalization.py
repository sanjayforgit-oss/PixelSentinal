"""
PixelSentinel Normalization Pipeline

Converts raw Landsat Collection 2 Level-2 digital numbers into
physically meaningful reflectance and temperature values before
normalizing them for Pix2Pix training.

Author
------
PixelSentinel Team

Module
------
preprocessing.normalization
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from preprocessing.config import (
    INPUT_BANDS,
    TARGET_BANDS,
    NORMALIZED_DTYPE,
    NORMALIZED_INPUT_DIR,
    NORMALIZED_TARGET_DIR,
    PROCESSED_DATASET_DIR,
    REFLECTANCE_SCALE,
    REFLECTANCE_OFFSET,
    THERMAL_SCALE,
    THERMAL_OFFSET,
    REFLECTANCE_CLIP,
    THERMAL_CLIP,
)

from preprocessing.logger import get_logger
from preprocessing.resource_monitor import (
    monitor_progress,
)
logger = get_logger(__name__)


INPUT_DIRECTORY = (
    PROCESSED_DATASET_DIR /
    "input"
)

TARGET_DIRECTORY = (
    PROCESSED_DATASET_DIR /
    "target"
)


# ==========================================================
# Result
# ==========================================================


@dataclass(slots=True)
class NormalizationResult:
    """
    Result of one scene normalization.
    """

    filename: str

    success: bool

    output_path: Path | None

    message: str


# ==========================================================
# Normalizer
# ==========================================================


class GeoTIFFNormalizer:
    """
    Production-grade Landsat GeoTIFF normalizer.
    """

    def __init__(self) -> None:

        self.input_directory = INPUT_DIRECTORY

        self.target_directory = TARGET_DIRECTORY

        self.output_input_directory = NORMALIZED_INPUT_DIR

        self.output_target_directory = NORMALIZED_TARGET_DIR

    # ======================================================
    # IO
    # ======================================================

    @staticmethod
    def open_scene(
        scene_path: Path,
    ) -> rasterio.DatasetReader:
        """
        Open GeoTIFF.
        """

        return rasterio.open(scene_path)

    @staticmethod
    def create_profile(
        dataset: rasterio.DatasetReader,
    ) -> dict[str, Any]:
        """
        Create output profile.
        """

        profile = dataset.profile.copy()

        profile.update(
            dtype=NORMALIZED_DTYPE,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        return profile

    @staticmethod
    def output_path(
        input_path: Path,
        destination_directory: Path,
    ) -> Path:
        """
        Output filename.
        """

        return (
            destination_directory /
            input_path.name
        )

    @staticmethod
    def already_processed(
        output_path: Path,
    ) -> bool:
        """
        Skip previously processed files.
        """

        return output_path.exists()

    @staticmethod
    def save_geotiff(
        output_path: Path,
        image: np.ndarray,
        profile: dict[str, Any],
        descriptions: list[str],
    ) -> None:
        """
        Save normalized GeoTIFF.
        """

        profile = profile.copy()

        profile.update(
            count=image.shape[0],
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as destination:

            destination.write(
                image.astype(
                    np.float32
                )
            )

            destination.descriptions = tuple(
                descriptions
            )

        logger.info(
            "Saved %s",
            output_path.name,
        )

    # ======================================================
    # Reflectance Normalization
    # ======================================================

    @staticmethod
    def normalize_reflectance(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert Landsat Surface Reflectance Digital Numbers
        into physical reflectance values.

        Formula
        -------
        Reflectance = DN × 0.0000275 - 0.2
        """

        image = image.astype(np.float32)

        image = (
            image * REFLECTANCE_SCALE
        ) + REFLECTANCE_OFFSET

        image = np.clip(
            image,
            REFLECTANCE_CLIP[0],
            REFLECTANCE_CLIP[1],
        )

        return image

    # ======================================================
    # Thermal Normalization
    # ======================================================

    @staticmethod
    def normalize_thermal(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert Landsat Surface Temperature Digital Numbers
        into Kelvin.

        Formula
        -------
        Temperature = DN × 0.00341802 + 149.0
        """

        image = image.astype(np.float32)

        image = (
            image * THERMAL_SCALE
        ) + THERMAL_OFFSET

        image = np.clip(
            image,
            THERMAL_CLIP[0],
            THERMAL_CLIP[1],
        )

        #
        # Normalize temperature to [0, 1]
        #

        image = (
            image - THERMAL_CLIP[0]
        ) / (
            THERMAL_CLIP[1] - THERMAL_CLIP[0]
        )

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        return image.astype(np.float32)

    # ======================================================
    # Input Normalization
    # ======================================================

    def normalize_input(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize the four input bands.

        Band Order
        ----------
        SR_B5
        SR_B6
        SR_B7
        ST_B10
        """

        output = np.empty_like(
            image,
            dtype=np.float32,
        )

        #
        # Reflectance
        #

        output[0] = self.normalize_reflectance(
            image[0]
        )

        output[1] = self.normalize_reflectance(
            image[1]
        )

        output[2] = self.normalize_reflectance(
            image[2]
        )

        #
        # Thermal
        #

        output[3] = self.normalize_thermal(
            image[3]
        )

        return output

    # ======================================================
    # Target Normalization
    # ======================================================

    def normalize_target(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize RGB target bands.

        Band Order
        ----------
        SR_B4
        SR_B3
        SR_B2
        """

        output = np.empty_like(
            image,
            dtype=np.float32,
        )

        for band in range(
            image.shape[0]
        ):

            output[band] = (
                self.normalize_reflectance(
                    image[band]
                )
            )

        return output

    # ======================================================
    # Scene Normalization
    # ======================================================

    def normalize_scene(
        self,
        dataset: rasterio.DatasetReader,
        is_input: bool,
    ) -> np.ndarray:
        """
        Normalize an entire GeoTIFF.
        """

        image = dataset.read()

        if is_input:

            return self.normalize_input(
                image
            )

        return self.normalize_target(
            image
        )

    # ======================================================
    # File Processing
    # ======================================================

    def process_file(
        self,
        input_file: Path,
        output_directory: Path,
        descriptions: list[str],
        is_input: bool,
    ) -> NormalizationResult:
        """
        Normalize a single GeoTIFF.
        """

        output_path = self.output_path(
            input_file,
            output_directory,
        )

        if self.already_processed(output_path):

            logger.info(
                "Skipping %s (already normalized)",
                input_file.name,
            )

            return NormalizationResult(
                filename=input_file.name,
                success=True,
                output_path=output_path,
                message="Already processed",
            )

        try:

            with self.open_scene(input_file) as dataset:

                normalized = self.normalize_scene(
                    dataset,
                    is_input=is_input,
                )

                profile = self.create_profile(
                    dataset,
                )

                self.save_geotiff(
                    output_path=output_path,
                    image=normalized,
                    profile=profile,
                    descriptions=descriptions,
                )

            return NormalizationResult(
                filename=input_file.name,
                success=True,
                output_path=output_path,
                message="Success",
            )

        except Exception as exc:

            logger.exception(
                "Failed processing %s",
                input_file.name,
            )

            return NormalizationResult(
                filename=input_file.name,
                success=False,
                output_path=None,
                message=str(exc),
            )

    # ======================================================
    # Dataset Processing
    # ======================================================

    def process_directory(
        self,
        input_directory: Path,
        output_directory: Path,
        descriptions: list[str],
        is_input: bool,
    ) -> list[NormalizationResult]:
        """
        Normalize an entire directory.
        """

        files = sorted(
            input_directory.glob("*.tif")
        )

        logger.info(
            "=" * 80,
        )

        logger.info(
            "Processing directory: %s",
            input_directory,
        )

        logger.info(
            "Files Found : %d",
            len(files),
        )

        results: list[NormalizationResult] = []

        total_files = len(files)

        for index, file in enumerate(files, start=1):

            monitor_progress(
                current=index,
                total=total_files,
                every=100,
            )

            result = self.process_file(
                input_file=file,
                output_directory=output_directory,
                descriptions=descriptions,
                is_input=is_input,
            )

            results.append(result)
        return results
    # ======================================================
    # Complete Dataset
    # ======================================================

    def process_dataset(
        self,
    ) -> list[NormalizationResult]:
        """
        Normalize the complete PixelSentinel dataset.
        """

        logger.info(
            "=" * 80,
        )

        logger.info(
            "Starting Landsat Normalization Pipeline",
        )

        logger.info(
            "=" * 80,
        )

        input_results = self.process_directory(
            input_directory=self.input_directory,
            output_directory=self.output_input_directory,
            descriptions=INPUT_BANDS,
            is_input=True,
        )

        target_results = self.process_directory(
            input_directory=self.target_directory,
            output_directory=self.output_target_directory,
            descriptions=TARGET_BANDS,
            is_input=False,
        )

        return input_results + target_results


# ==========================================================
# Summary
# ==========================================================


def print_summary(
    results: list[NormalizationResult],
    execution_time: float,
) -> None:
    """
    Print normalization summary.
    """

    total = len(results)

    successful = sum(
        result.success
        for result in results
    )

    failed = total - successful

    logger.info("=" * 80)
    logger.info("NORMALIZATION SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Files Processed : %d",
        total,
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
        "Execution Time  : %.2f seconds",
        execution_time,
    )

    if failed:

        logger.info("-" * 80)
        logger.info("FAILED FILES")
        logger.info("-" * 80)

        for result in results:

            if not result.success:

                logger.info(
                    "%s -> %s",
                    result.filename,
                    result.message,
                )

    logger.info("=" * 80)


# ==========================================================
# Output Validation
# ==========================================================


def validate_outputs() -> None:
    """
    Verify that normalized dataset exists.
    """

    input_count = len(
        list(
            NORMALIZED_INPUT_DIR.glob("*.tif")
        )
    )

    target_count = len(
        list(
            NORMALIZED_TARGET_DIR.glob("*.tif")
        )
    )

    logger.info(
        "Normalized Input Files  : %d",
        input_count,
    )

    logger.info(
        "Normalized Target Files : %d",
        target_count,
    )

    if input_count != target_count:

        logger.warning(
            "Input and Target counts do not match."
        )

    else:

        logger.info(
            "Normalization dataset verified."
        )

def normalize_dataset():
    """
    Pipeline entry point for normalization.
    """

    start_time = time.perf_counter()

    normalizer = GeoTIFFNormalizer()

    results = normalizer.process_dataset()

    elapsed = time.perf_counter() - start_time

    print_summary(
        results,
        elapsed,
    )

    validate_outputs()
    logger.info(
        "Execution Time : %.2f seconds",
        elapsed,
    )
    success = sum(r.success for r in results)
    failed = len(results) - success

    return {
        "success": failed == 0,
        "processed": success,
        "failed": failed,
        "execution_time": elapsed,
    }


# ==========================================================
# Main
# ==========================================================
def process_dataset():
    return normalize_dataset()

def main() -> None:
    """
    Execute the complete normalization pipeline.
    """

    logger.info("=" * 80)
    logger.info("PixelSentinel Landsat Normalization")
    logger.info("=" * 80)

    start_time = time.perf_counter()

    normalizer = GeoTIFFNormalizer()

    results = normalizer.process_dataset()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print_summary(
        results,
        elapsed,
    )

    validate_outputs()


# ==========================================================
# Entry Point
# ==========================================================


if __name__ == "__main__":

    main()