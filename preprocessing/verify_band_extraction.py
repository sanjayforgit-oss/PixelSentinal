"""
PixelSentinel Band Extraction Verification

Verifies that extracted input and target GeoTIFFs exactly match the
original raw Landsat scene.

Checks
------
- File existence
- Band count
- Dimensions
- CRS
- Affine transform
- Band descriptions
- Data type
- Pixel equality
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from preprocessing.config import (
    RAW_DATASET_DIR,
    PROCESSED_DATASET_DIR,
    INPUT_BANDS,
    TARGET_BANDS,
)

from preprocessing.logger import get_logger


logger = get_logger(__name__)


INPUT_DIR = PROCESSED_DATASET_DIR / "input"
TARGET_DIR = PROCESSED_DATASET_DIR / "target"


@dataclass(slots=True)
class VerificationResult:

    filename: str

    success: bool

    message: str


class BandExtractionVerifier:

    def __init__(
        self,
        raw_dir: Path = RAW_DATASET_DIR,
        input_dir: Path = INPUT_DIR,
        target_dir: Path = TARGET_DIR,
    ) -> None:

        self.raw_dir = raw_dir
        self.input_dir = input_dir
        self.target_dir = target_dir

    @staticmethod
    def compare_metadata(
        raw: rasterio.DatasetReader,
        extracted: rasterio.DatasetReader,
    ) -> None:

        if raw.crs != extracted.crs:
            raise ValueError("CRS mismatch")

        if raw.transform != extracted.transform:
            raise ValueError("Affine transform mismatch")

        if raw.width != extracted.width:
            raise ValueError("Width mismatch")

        if raw.height != extracted.height:
            raise ValueError("Height mismatch")

        if raw.dtypes[0] != extracted.dtypes[0]:
            raise ValueError("Data type mismatch")

    @staticmethod
    def compare_descriptions(
        extracted: rasterio.DatasetReader,
        expected: list[str],
    ) -> None:

        descriptions = list(extracted.descriptions)

        if descriptions != expected:
            raise ValueError(
                f"Band descriptions incorrect.\n"
                f"Expected: {expected}\n"
                f"Found: {descriptions}"
            )

    @staticmethod
    def compare_pixels(
        raw_array: np.ndarray,
        extracted_array: np.ndarray,
    ) -> None:

        if not np.array_equal(raw_array, extracted_array):
            raise ValueError("Pixel values differ")

    def verify_scene(
        self,
        raw_scene: Path,
    ) -> VerificationResult:

        logger.info("-" * 80)

        logger.info(
            "Verifying %s",
            raw_scene.name,
        )

        input_file = (
            self.input_dir /
            f"{raw_scene.stem}_input.tif"
        )

        target_file = (
            self.target_dir /
            f"{raw_scene.stem}_target.tif"
        )

        if not input_file.exists():

            return VerificationResult(
                raw_scene.name,
                False,
                "Missing input file",
            )

        if not target_file.exists():

            return VerificationResult(
                raw_scene.name,
                False,
                "Missing target file",
            )

        try:

            with rasterio.open(raw_scene) as raw, \
                 rasterio.open(input_file) as input_ds, \
                 rasterio.open(target_file) as target_ds:

                if input_ds.count != 4:
                    raise ValueError("Input must contain 4 bands")

                if target_ds.count != 3:
                    raise ValueError("Target must contain 3 bands")

                self.compare_metadata(raw, input_ds)

                self.compare_metadata(raw, target_ds)

                self.compare_descriptions(
                    input_ds,
                    INPUT_BANDS,
                )

                self.compare_descriptions(
                    target_ds,
                    TARGET_BANDS,
                )

                raw_descriptions = list(raw.descriptions)

                input_indexes = [
                    raw_descriptions.index(band) + 1
                    for band in INPUT_BANDS
                ]

                target_indexes = [
                    raw_descriptions.index(band) + 1
                    for band in TARGET_BANDS
                ]

                raw_input = raw.read(input_indexes)

                raw_target = raw.read(target_indexes)

                input_array = input_ds.read()

                target_array = target_ds.read()

                self.compare_pixels(
                    raw_input,
                    input_array,
                )

                self.compare_pixels(
                    raw_target,
                    target_array,
                )

                logger.info("PASS")

                return VerificationResult(
                    raw_scene.name,
                    True,
                    "PASS",
                )

        except Exception as exc:

            logger.exception(raw_scene.name)

            return VerificationResult(
                raw_scene.name,
                False,
                str(exc),
            )

# ==========================================================
# Dataset Verification
# ==========================================================


def verify_dataset() -> list[VerificationResult]:
    """
    Verify every extracted scene.
    """

    verifier = BandExtractionVerifier()

    raw_files = sorted(
        verifier.raw_dir.glob("*.tif")
    )

    logger.info("=" * 80)
    logger.info("Starting Band Extraction Verification")
    logger.info("=" * 80)
    logger.info(
        "Scenes Found : %d",
        len(raw_files),
    )

    results: list[VerificationResult] = []

    for raw_scene in raw_files:

        result = verifier.verify_scene(raw_scene)

        results.append(result)

    return results


# ==========================================================
# Summary
# ==========================================================


def print_summary(
    results: list[VerificationResult],
) -> None:
    """
    Print verification summary.
    """

    total = len(results)

    passed = sum(
        result.success
        for result in results
    )

    failed = total - passed

    logger.info("=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Scenes Verified : %d",
        total,
    )

    logger.info(
        "Passed          : %d",
        passed,
    )

    logger.info(
        "Failed          : %d",
        failed,
    )

    if failed:

        logger.info("-" * 80)

        logger.info("Failed Scenes")

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
# Main
# ==========================================================


def main() -> None:

    import time

    start = time.perf_counter()

    results = verify_dataset()

    elapsed = time.perf_counter() - start

    print_summary(results)

    logger.info(
        "Execution Time : %.2f seconds",
        elapsed,
    )


if __name__ == "__main__":

    main()