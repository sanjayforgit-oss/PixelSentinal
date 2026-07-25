"""
PixelSentinel Image Registration Verification

This module verifies that every normalized input GeoTIFF and its
corresponding normalized target GeoTIFF remain perfectly
co-registered.

Unlike feature-based registration (ECC/SIFT/ORB), Landsat Collection-2
products are already orthorectified and co-registered.

Therefore this module performs scientific verification instead of
geometric correction.

Checks
------
✓ CRS
✓ Affine Transform
✓ Resolution
✓ Bounds
✓ Width
✓ Height
✓ Band Count
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path

import rasterio

from preprocessing.config import (
    NORMALIZED_INPUT_DIR,
    NORMALIZED_TARGET_DIR,
    INPUT_BANDS,
    TARGET_BANDS,
)

from preprocessing.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# Result
# ==========================================================


@dataclass(slots=True)
class RegistrationResult:
    """
    Registration verification result.
    """

    filename: str

    success: bool

    message: str


# ==========================================================
# Registration Verifier
# ==========================================================


class ImageRegistration:
    """
    Verify geospatial registration between paired
    normalized GeoTIFFs.
    """

    def __init__(
        self,
        input_directory: Path = NORMALIZED_INPUT_DIR,
        target_directory: Path = NORMALIZED_TARGET_DIR,
    ) -> None:

        self.input_directory = input_directory

        self.target_directory = target_directory

    # ======================================================
    # CRS
    # ======================================================

    @staticmethod
    def compare_crs(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify CRS.
        """

        if input_dataset.crs != target_dataset.crs:

            raise ValueError(
                "CRS mismatch."
            )

    # ======================================================
    # Transform
    # ======================================================

    @staticmethod
    def compare_transform(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify affine transform.
        """

        if input_dataset.transform != target_dataset.transform:

            raise ValueError(
                "Affine transform mismatch."
            )

    # ======================================================
    # Resolution
    # ======================================================

    @staticmethod
    def compare_resolution(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify pixel resolution.
        """

        if input_dataset.res != target_dataset.res:

            raise ValueError(
                "Resolution mismatch."
            )

    # ======================================================
    # Dimensions
    # ======================================================

    @staticmethod
    def compare_dimensions(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify image dimensions.
        """

        if input_dataset.width != target_dataset.width:

            raise ValueError(
                "Width mismatch."
            )

        if input_dataset.height != target_dataset.height:

            raise ValueError(
                "Height mismatch."
            )

    # ======================================================
    # Bounds
    # ======================================================

    @staticmethod
    def compare_bounds(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify spatial bounds.
        """

        if input_dataset.bounds != target_dataset.bounds:

            raise ValueError(
                "Spatial bounds mismatch."
            )

    # ======================================================
    # Band Count
    # ======================================================

    @staticmethod
    def compare_band_count(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify the expected number of bands.
        """

        if input_dataset.count != len(INPUT_BANDS):

            raise ValueError(
                f"Input image has {input_dataset.count} bands "
                f"(expected {len(INPUT_BANDS)})."
            )

        if target_dataset.count != len(TARGET_BANDS):

            raise ValueError(
                f"Target image has {target_dataset.count} bands "
                f"(expected {len(TARGET_BANDS)})."
            )

    # ======================================================
    # Band Descriptions
    # ======================================================

    @staticmethod
    def compare_band_descriptions(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify band descriptions.
        """

        input_descriptions = list(
            input_dataset.descriptions
        )

        target_descriptions = list(
            target_dataset.descriptions
        )

        if input_descriptions != INPUT_BANDS:

            raise ValueError(
                "Input band descriptions mismatch.\n"
                f"Expected : {INPUT_BANDS}\n"
                f"Found    : {input_descriptions}"
            )

        if target_descriptions != TARGET_BANDS:

            raise ValueError(
                "Target band descriptions mismatch.\n"
                f"Expected : {TARGET_BANDS}\n"
                f"Found    : {target_descriptions}"
            )

    # ======================================================
    # Logging
    # ======================================================

    @staticmethod
    def log_registration(
        input_dataset: rasterio.DatasetReader,
        target_dataset: rasterio.DatasetReader,
        filename: str,
    ) -> None:
        """
        Log registration information.
        """

        logger.debug(
            (
                "%s | CRS=%s | Resolution=%s | "
                "Size=%dx%d"
            ),
            filename,
            input_dataset.crs,
            input_dataset.res,
            input_dataset.width,
            input_dataset.height,
        )

    # ======================================================
    # Pair Verification
    # ======================================================

    def verify_scene(
        self,
        input_path: Path,
        target_path: Path,
    ) -> RegistrationResult:
        """
        Verify one input-target GeoTIFF pair.
        """

        try:

            with rasterio.open(input_path) as input_dataset, \
                 rasterio.open(target_path) as target_dataset:

                self.compare_crs(
                    input_dataset,
                    target_dataset,
                )

                self.compare_transform(
                    input_dataset,
                    target_dataset,
                )

                self.compare_resolution(
                    input_dataset,
                    target_dataset,
                )

                self.compare_dimensions(
                    input_dataset,
                    target_dataset,
                )

                self.compare_bounds(
                    input_dataset,
                    target_dataset,
                )

                self.compare_band_count(
                    input_dataset,
                    target_dataset,
                )

                self.compare_band_descriptions(
                    input_dataset,
                    target_dataset,
                )

                self.log_registration(
                    input_dataset,
                    target_dataset,
                    input_path.name,
                )

                return RegistrationResult(
                    filename=input_path.name,
                    success=True,
                    message="PASS",
                )

        except Exception as exc:

            logger.exception(
                "Registration failed: %s",
                input_path.name,
            )

            return RegistrationResult(
                filename=input_path.name,
                success=False,
                message=str(exc),
            )

    # ======================================================
    # Dataset Verification
    # ======================================================

    def verify_dataset(
        self,
    ) -> list[RegistrationResult]:
        """
        Verify registration for the complete normalized dataset.
        """

        input_files = sorted(
            self.input_directory.glob("*.tif")
        )

        target_files = sorted(
            self.target_directory.glob("*.tif")
        )

        if len(input_files) != len(target_files):

            raise RuntimeError(
                "Input/Target file count mismatch.\n"
                f"Input : {len(input_files)}\n"
                f"Target: {len(target_files)}"
            )

        logger.info(
            "Found %d input files.",
            len(input_files),
        )

        logger.info(
            "Found %d target files.",
            len(target_files),
        )

        results: list[RegistrationResult] = []

        total = len(input_files)

        for index, (input_path, target_path) in enumerate(
            zip(input_files, target_files),
            start=1,
        ):

            #
            # Verify matching filenames
            #

            input_name = input_path.stem.replace(
                "_input",
                "",
            )

            target_name = target_path.stem.replace(
                "_target",
                "",
            )

            if input_name != target_name:

                results.append(
                    RegistrationResult(
                        filename=input_path.name,
                        success=False,
                        message="Input/Target filename mismatch.",
                    )
                )

                continue

            #
            # Progress logging
            #

            if (
                index == 1
                or index % 100 == 0
                or index == total
            ):

                logger.info(
                    "Verified %d / %d scene pairs...",
                    index,
                    total,
                )

            results.append(
                self.verify_scene(
                    input_path,
                    target_path,
                )
            )

        return results

    # ======================================================
    # Registration Summary
    # ======================================================

    def run(
        self,
    ) -> None:
        """
        Execute registration verification.
        """

        logger.info("=" * 80)
        logger.info(
            "VERIFYING IMAGE REGISTRATION"
        )
        logger.info("=" * 80)

        start = time.perf_counter()

        results = self.verify_dataset()

        passed = sum(
            result.success
            for result in results
        )

        failed = len(results) - passed

        elapsed = time.perf_counter() - start

        logger.info("=" * 80)
        logger.info(
            "IMAGE REGISTRATION SUMMARY"
        )
        logger.info("=" * 80)

        logger.info(
            "Scene Pairs Checked : %d",
            len(results),
        )

        logger.info(
            "Passed              : %d",
            passed,
        )

        logger.info(
            "Failed              : %d",
            failed,
        )

        logger.info("=" * 80)

        logger.info(
            "Execution Time : %.2f seconds",
            elapsed,
        )

        if failed == 0:

            logger.info(
                "All image pairs are perfectly registered."
            )

        else:

            logger.warning(
                "Registration verification completed with failures."
            )


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    """
    Program entry point.
    """

    registration = ImageRegistration()

    registration.run()


if __name__ == "__main__":
    main()
