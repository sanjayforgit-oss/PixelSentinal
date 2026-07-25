"""
PixelSentinel Normalization Verification

Verifies the integrity of normalized Landsat GeoTIFFs.

Checks
------
- File existence
- Band count
- CRS
- Transform
- Dimensions
- Data type
- Band descriptions
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from preprocessing.config import (
    INPUT_BANDS,
    TARGET_BANDS,
    NORMALIZED_INPUT_DIR,
    NORMALIZED_TARGET_DIR,
    PROCESSED_DATASET_DIR,
    NORMALIZED_DTYPE,
)

from preprocessing.logger import get_logger


logger = get_logger(__name__)


INPUT_DIR = PROCESSED_DATASET_DIR / "input"
TARGET_DIR = PROCESSED_DATASET_DIR / "target"


# ==========================================================
# Result
# ==========================================================


@dataclass(slots=True)
class VerificationResult:
    """
    Verification result for one normalized GeoTIFF.
    """

    filename: str

    success: bool

    message: str


# ==========================================================
# Verifier
# ==========================================================


class NormalizationVerifier:
    """
    Verify normalized Landsat GeoTIFFs.
    """

    def __init__(
        self,
        input_directory: Path = NORMALIZED_INPUT_DIR,
        target_directory: Path = NORMALIZED_TARGET_DIR,
    ) -> None:

        self.input_directory = input_directory

        self.target_directory = target_directory

    # ======================================================
    # Metadata
    # ======================================================

    @staticmethod
    def compare_metadata(
        original: rasterio.DatasetReader,
        normalized: rasterio.DatasetReader,
    ) -> None:
        """
        Verify metadata consistency.
        """

        if original.crs != normalized.crs:
            raise ValueError("CRS mismatch")

        if original.transform != normalized.transform:
            raise ValueError("Affine transform mismatch")

        if original.width != normalized.width:
            raise ValueError("Width mismatch")

        if original.height != normalized.height:
            raise ValueError("Height mismatch")

    # ======================================================
    # Band Count
    # ======================================================

    @staticmethod
    def verify_band_count(
        dataset: rasterio.DatasetReader,
        expected: int,
    ) -> None:
        """
        Verify number of bands.
        """

        if dataset.count != expected:

            raise ValueError(
                f"Expected {expected} bands "
                f"but found {dataset.count}"
            )

    # ======================================================
    # Data Type
    # ======================================================

    @staticmethod
    def verify_dtype(
        dataset: rasterio.DatasetReader,
    ) -> None:
        """
        Verify normalized dtype.
        """

        dtype = dataset.dtypes[0]

        if dtype != NORMALIZED_DTYPE:

            raise ValueError(
                f"Expected {NORMALIZED_DTYPE} "
                f"but found {dtype}"
            )

    # ======================================================
    # Band Descriptions
    # ======================================================

    @staticmethod
    def verify_descriptions(
        dataset: rasterio.DatasetReader,
        expected: list[str],
    ) -> None:
        """
        Verify band descriptions.
        """

        descriptions = list(
            dataset.descriptions
        )

        if descriptions != expected:

            raise ValueError(
                "Band descriptions do not match.\n"
                f"Expected : {expected}\n"
                f"Found    : {descriptions}"
            )

    # ======================================================
    # Value Range
    # ======================================================

    @staticmethod
    def verify_value_range(
        image: np.ndarray,
        is_input: bool,
    ) -> None:
        """
        Verify normalized value ranges.

        Input
        -----
        Bands 1-3 : Reflectance [0,1]
        Band 4    : Thermal     [0,1]

        Target
        ------
        All bands : Reflectance [0,1]
        """

        tolerance = 1e-6

        if is_input:

            #
            # Reflectance
            #

            for band in range(3):

                minimum = image[band].min()
                maximum = image[band].max()

                if (
                    minimum < -tolerance
                    or maximum > 1.0 + tolerance
                ):

                    raise ValueError(
                        f"Reflectance band {band + 1} "
                        f"outside [0,1]. "
                        f"Min={minimum:.6f}, "
                        f"Max={maximum:.6f}"
                    )

            #
            # Thermal
            #

            minimum = image[3].min()
            maximum = image[3].max()

            if (
                minimum < -tolerance
                or maximum > 1.0 + tolerance
            ):

                raise ValueError(
                    "Thermal band outside [0,1]. "
                    f"Min={minimum:.6f}, "
                    f"Max={maximum:.6f}"
                )

        else:

            for band in range(image.shape[0]):

                minimum = image[band].min()
                maximum = image[band].max()

                if (
                    minimum < -tolerance
                    or maximum > 1.0 + tolerance
                ):

                    raise ValueError(
                        f"Target band {band + 1} "
                        f"outside [0,1]. "
                        f"Min={minimum:.6f}, "
                        f"Max={maximum:.6f}"
                    )

    # ======================================================
    # NaN / Inf
    # ======================================================

    @staticmethod
    def verify_numeric_values(
        image: np.ndarray,
    ) -> None:
        """
        Verify finite numeric values.
        """

        if np.isnan(image).any():

            raise ValueError(
                "NaN values detected."
            )

        if np.isinf(image).any():

            raise ValueError(
                "Infinite values detected."
            )

    # ======================================================
    # Statistics
    # ======================================================

    @staticmethod
    def log_statistics(
        image: np.ndarray,
        filename: str,
    ) -> None:
        """
        Log dataset statistics.
        """

        logger.debug(
            "%s | shape=%s | min=%.6f | max=%.6f | mean=%.6f",
            filename,
            image.shape,
            float(image.min()),
            float(image.max()),
            float(image.mean()),
        )

    # ======================================================
    # Scene Verification
    # ======================================================

    def verify_scene(
        self,
        original_path: Path,
        normalized_path: Path,
        expected_bands: int,
        expected_descriptions: list[str],
        is_input: bool,
    ) -> VerificationResult:
        """
        Verify one normalized GeoTIFF.
        """

        try:

            with rasterio.open(original_path) as original, \
                 rasterio.open(normalized_path) as normalized:

                self.compare_metadata(
                    original,
                    normalized,
                )

                self.verify_band_count(
                    normalized,
                    expected_bands,
                )

                self.verify_dtype(
                    normalized,
                )

                self.verify_descriptions(
                    normalized,
                    expected_descriptions,
                )

                image = normalized.read()

                self.verify_numeric_values(
                    image,
                )

                self.verify_value_range(
                    image,
                    is_input=is_input,
                )

                self.log_statistics(
                    image,
                    normalized_path.name,
                )

                return VerificationResult(
                    filename=normalized_path.name,
                    success=True,
                    message="PASS",
                )

        except Exception as exc:

            logger.exception(
                normalized_path.name,
            )

            return VerificationResult(
                filename=normalized_path.name,
                success=False,
                message=str(exc),
            )

    # ======================================================
    # Directory Verification
    # ======================================================

    def verify_directory(
        self,
        original_directory: Path,
        normalized_directory: Path,
        expected_bands: int,
        expected_descriptions: list[str],
        is_input: bool,
    ) -> list[VerificationResult]:
        """
        Verify every normalized GeoTIFF in a directory.
        """

        results: list[VerificationResult] = []

        original_files = sorted(
            original_directory.glob("*.tif")
        )

        normalized_files = sorted(
            normalized_directory.glob("*.tif")
        )

        if len(original_files) != len(normalized_files):

            raise RuntimeError(
                "Directory size mismatch.\n"
                f"Original   : {len(original_files)}\n"
                f"Normalized : {len(normalized_files)}"
            )

        for index, (original_path, normalized_path) in enumerate(
            zip(original_files, normalized_files),
            start=1,
        ):

            if index % 100 == 0:
                logger.info(
                    "Verified %d / %d files...",
                    index,
                    len(original_files),
                )

            result = self.verify_scene(
                original_path=original_path,
                normalized_path=normalized_path,
                expected_bands=expected_bands,
                expected_descriptions=expected_descriptions,
                is_input=is_input,
            )

            results.append(result)

        return results

    # ======================================================
    # Full Verification
    # ======================================================

    def verify(self) -> None:
        """
        Verify the complete normalized dataset.
        """

        logger.info("=" * 80)
        logger.info("VERIFYING NORMALIZED DATASET")
        logger.info("=" * 80)

        start = time.perf_counter()

        input_results = self.verify_directory(
            original_directory=INPUT_DIR,
            normalized_directory=self.input_directory,
            expected_bands=len(INPUT_BANDS),
            expected_descriptions=INPUT_BANDS,
            is_input=True,
        )

        target_results = self.verify_directory(
            original_directory=TARGET_DIR,
            normalized_directory=self.target_directory,
            expected_bands=len(TARGET_BANDS),
            expected_descriptions=TARGET_BANDS,
            is_input=False,
        )

        results = input_results + target_results

        passed = sum(
            result.success
            for result in results
        )

        failed = len(results) - passed

        elapsed = time.perf_counter() - start

        logger.info("=" * 80)
        logger.info("NORMALIZATION VERIFICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            "Files Verified : %d",
            len(results),
        )
        logger.info(
            "Passed          : %d",
            passed,
        )
        logger.info(
            "Failed          : %d",
            failed,
        )
        logger.info("=" * 80)
        logger.info(
            "Execution Time : %.2f seconds",
            elapsed,
        )

        if failed == 0:

            logger.info(
                "Normalized dataset verification PASSED."
            )

        else:

            logger.warning(
                "Normalized dataset verification FAILED."
            )

# ==========================================================
# Main
# ==========================================================


def main() -> None:
    """
    Entry point.
    """

    verifier = NormalizationVerifier()

    verifier.verify()


if __name__ == "__main__":
    main()