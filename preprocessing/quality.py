"""
PixelSentinel Quality Validation Module

This module validates all raw Landsat GeoTIFF scenes before they enter
the preprocessing pipeline.

Validation includes:

- File readability
- GeoTIFF metadata
- CRS
- Affine transform
- Band count
- Band descriptions
- Data types
- Image dimensions
- Pixel statistics
- Missing values
- Empty bands

A CSV report is generated for every processed scene.

Author:
PixelSentinel Team

Project:
ISRO Bharatiya Antariksh Hackathon 2026
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine

from preprocessing.config import (
    RAW_DATASET_DIR,
    METADATA_DIR,
    QUALITY_REPORT_FILE,
    EXPECTED_TOTAL_BANDS,
    EXPECTED_BAND_ORDER,
    SUPPORTED_FILE_EXTENSIONS,
)

from preprocessing.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# Dataclasses
# ==========================================================


@dataclass(slots=True)
class BandStatistics:
    """
    Statistics for a single raster band.
    """

    band_name: str

    minimum: float
    maximum: float
    mean: float
    std: float

    nan_pixels: int
    inf_pixels: int

    zero_pixels: int

    empty_band: bool


@dataclass(slots=True)
class QualityResult:
    """
    Validation result for a single GeoTIFF scene.
    """

    filename: str

    valid: bool

    validation_time: str

    width: int
    height: int

    band_count: int

    dtype: str

    crs: str

    transform: str

    errors: list[str]

    warnings: list[str]

    statistics: list[BandStatistics] | None = None


# ==========================================================
# Quality Checker
# ==========================================================


class QualityChecker:
    """
    Validates Landsat GeoTIFF scenes.

    Every scene is validated independently.

    The checker performs metadata validation,
    pixel validation and statistical analysis.

    Results are returned as QualityResult.
    """

    def __init__(
        self,
        dataset_dir: Path = RAW_DATASET_DIR,
        report_path: Path = QUALITY_REPORT_FILE,
    ) -> None:

        self.dataset_dir = dataset_dir
        self.report_path = report_path

        self.results: list[QualityResult] = []

        logger.info("QualityChecker initialized.")
        logger.info("Dataset directory : %s", dataset_dir)
        logger.info("Report path       : %s", report_path)

    # ======================================================
    # Helper Methods
    # ======================================================

    @staticmethod
    def _is_supported_file(path: Path) -> bool:
        """
        Check whether a file extension is supported.
        """

        return path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS

    @staticmethod
    def _current_timestamp() -> str:
        """
        Return current timestamp.
        """

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _validate_affine(transform: Affine) -> bool:
        """
        Validate affine transform.

        Returns
        -------
        bool
            True if transform is valid.
        """

        if transform is None:
            return False

        if not isinstance(transform, Affine):
            return False

        if transform.a == 0:
            return False

        if transform.e == 0:
            return False

        return True

    @staticmethod
    def _validate_crs(crs: Any) -> bool:
        """
        Validate coordinate reference system.
        """

        return crs is not None

    @staticmethod
    def _validate_dimensions(
        width: int,
        height: int,
    ) -> bool:
        """
        Validate image dimensions.
        """

        return width > 0 and height > 0

    @staticmethod
    def _validate_band_count(
        band_count: int,
    ) -> bool:
        """
        Validate total number of bands.
        """

        return band_count == EXPECTED_TOTAL_BANDS

    @staticmethod
    def _validate_dtype(
        dtypes: tuple[str, ...],
    ) -> bool:
        """
        Ensure every band has the same dtype.
        """

        return len(set(dtypes)) == 1

    @staticmethod
    def _validate_band_descriptions(
        descriptions: tuple[str | None, ...],
    ) -> bool:
        """
        Validate exported band order.
        """

        clean = [
            item.strip() if item is not None else ""
            for item in descriptions
        ]

        return clean == EXPECTED_BAND_ORDER

    # ======================================================
    # File Opening
    # ======================================================

    def open_scene(
        self,
        scene_path: Path,
    ) -> rasterio.DatasetReader:
        """
        Open a GeoTIFF scene safely.

        Parameters
        ----------
        scene_path : Path

        Returns
        -------
        rasterio.DatasetReader
        """

        if not scene_path.exists():
            raise FileNotFoundError(scene_path)

        if not self._is_supported_file(scene_path):
            raise ValueError(
                f"Unsupported file type: {scene_path.suffix}"
            )

        try:

            dataset = rasterio.open(scene_path)

            logger.info(
                "Opened scene %s",
                scene_path.name,
            )

            return dataset

        except RasterioIOError as exc:

            logger.exception(
                "Rasterio failed to read %s",
                scene_path,
            )

            raise exc

    # ======================================================
    # Metadata Validation
    # ======================================================

    def validate_metadata(
        self,
        dataset: rasterio.DatasetReader,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """
        Validate dataset metadata.

        Parameters
        ----------
        dataset
            Open rasterio dataset.

        errors
            Error collector.

        warnings
            Warning collector.
        """

        if not self._validate_crs(dataset.crs):
            errors.append("Invalid CRS")

        if not self._validate_affine(dataset.transform):
            errors.append("Invalid affine transform")

        if not self._validate_dimensions(
            dataset.width,
            dataset.height,
        ):
            errors.append("Invalid image dimensions")

        if not self._validate_band_count(
            dataset.count,
        ):
            errors.append(
                f"Expected {EXPECTED_TOTAL_BANDS} bands "
                f"but found {dataset.count}"
            )

        if not self._validate_dtype(dataset.dtypes):
            errors.append(
                "Band data types are inconsistent."
            )

        if not self._validate_band_descriptions(
            dataset.descriptions,
        ):
            warnings.append(
                "Band descriptions do not match "
                "expected PixelSentinel ordering."
            )

    # ======================================================
    # Pixel Validation
    # ======================================================

    @staticmethod
    def _count_nan(array: np.ndarray) -> int:
        """
        Count NaN pixels.
        """

        return int(np.isnan(array).sum())

    @staticmethod
    def _count_inf(array: np.ndarray) -> int:
        """
        Count infinite pixels.
        """

        return int(np.isinf(array).sum())

    @staticmethod
    def _count_zero(array: np.ndarray) -> int:
        """
        Count zero-valued pixels.
        """

        return int(np.count_nonzero(array == 0))

    @staticmethod
    def _is_empty_band(array: np.ndarray) -> bool:
        """
        Determine whether an entire band contains only zeros.
        """

        return bool(np.all(array == 0))

    @staticmethod
    def _is_constant_band(array: np.ndarray) -> bool:
        """
        Determine whether every pixel has the same value.
        """

        return bool(np.min(array) == np.max(array))

    @staticmethod
    def _nodata_percentage(
        array: np.ndarray,
        nodata: float | int | None,
    ) -> float:
        """
        Calculate percentage of NoData pixels.
        """

        if nodata is None:
            return 0.0

        nodata_pixels = np.count_nonzero(array == nodata)

        return (
            nodata_pixels / array.size
        ) * 100.0

    # ======================================================
    # Band Statistics
    # ======================================================

    def compute_band_statistics(
        self,
        array: np.ndarray,
        band_name: str,
    ) -> BandStatistics:
        """
        Compute statistics for a raster band.
        """

        array = array.astype(np.float32)

        nan_pixels = self._count_nan(array)
        inf_pixels = self._count_inf(array)
        zero_pixels = self._count_zero(array)

        empty_band = self._is_empty_band(array)

        valid_pixels = array[
            np.isfinite(array)
        ]

        if valid_pixels.size == 0:

            minimum = 0.0
            maximum = 0.0
            mean = 0.0
            std = 0.0

        else:

            minimum = float(np.min(valid_pixels))
            maximum = float(np.max(valid_pixels))
            mean = float(np.mean(valid_pixels))
            std = float(np.std(valid_pixels))

        return BandStatistics(
            band_name=band_name,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            nan_pixels=nan_pixels,
            inf_pixels=inf_pixels,
            zero_pixels=zero_pixels,
            empty_band=empty_band,
        )

    # ======================================================
    # Pixel Validation
    # ======================================================

    def validate_pixels(
        self,
        dataset: rasterio.DatasetReader,
        errors: list[str],
        warnings: list[str],
    ) -> list[BandStatistics]:
        """
        Validate every raster band.

        Returns
        -------
        list[BandStatistics]
        """

        statistics: list[BandStatistics] = []

        for index in range(dataset.count):

            band_name = (
                dataset.descriptions[index]
                if dataset.descriptions[index]
                else f"Band_{index + 1}"
            )

            band = dataset.read(index + 1)

            stats = self.compute_band_statistics(
                band,
                band_name,
            )

            statistics.append(stats)

            if stats.nan_pixels > 0:

                errors.append(
                    f"{band_name}: "
                    f"{stats.nan_pixels} NaN pixels"
                )

            if stats.inf_pixels > 0:

                errors.append(
                    f"{band_name}: "
                    f"{stats.inf_pixels} Infinite pixels"
                )

            if stats.empty_band:

                errors.append(
                    f"{band_name}: Empty band"
                )

            if self._is_constant_band(
                band.astype(np.float32)
            ):

                warnings.append(
                    f"{band_name}: Constant pixel values"
                )

            nodata_percentage = self._nodata_percentage(
                band,
                dataset.nodata,
            )

            if nodata_percentage > 5:

                warnings.append(
                    f"{band_name}: "
                    f"{nodata_percentage:.2f}% NoData pixels"
                )

            if stats.minimum == stats.maximum:

                warnings.append(
                    f"{band_name}: "
                    "No dynamic range"
                )

        return statistics

    # ======================================================
    # Scene Validation
    # ======================================================

    def validate_scene(
        self,
        scene_path: Path,
    ) -> QualityResult:
        """
        Validate a single GeoTIFF scene.
        """

        logger.info(
            "Validating %s",
            scene_path.name,
        )

        errors: list[str] = []
        warnings: list[str] = []

        dataset = self.open_scene(scene_path)

        with dataset:

            self.validate_metadata(
                dataset,
                errors,
                warnings,
            )

            statistics = self.validate_pixels(
                dataset,
                errors,
                warnings,
            )

            result = QualityResult(
                filename=scene_path.name,
                valid=len(errors) == 0,
                validation_time=self._current_timestamp(),
                width=dataset.width,
                height=dataset.height,
                band_count=dataset.count,
                dtype=str(dataset.dtypes[0]),
                crs=str(dataset.crs),
                transform=str(dataset.transform),
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )

        self.results.append(result)

        logger.debug(
            "Validation completed with %d errors and %d warnings.",
            len(errors),
            len(warnings),
        )

        if result.valid:

            logger.info(
                "%s passed validation.",
                scene_path.name,
            )

        else:

            logger.warning(
                "%s failed validation.",
                scene_path.name,
            )

        return result

    # ======================================================
    # CSV Report
    # ======================================================

    def _flatten_result(
        self,
        result: QualityResult,
    ) -> dict[str, object]:
        """
        Convert a QualityResult into a CSV-compatible dictionary.
        """

        row: dict[str, object] = {
            "Filename": result.filename,
            "Valid": result.valid,
            "ValidationTime": result.validation_time,
            "Width": result.width,
            "Height": result.height,
            "BandCount": result.band_count,
            "DataType": result.dtype,
            "CRS": result.crs,
            "Transform": result.transform,
            "Errors": "; ".join(result.errors),
            "Warnings": "; ".join(result.warnings),
        }

        if result.statistics:

            for stat in result.statistics:

                prefix = stat.band_name

                row[f"{prefix}_Min"] = stat.minimum
                row[f"{prefix}_Max"] = stat.maximum
                row[f"{prefix}_Mean"] = stat.mean
                row[f"{prefix}_Std"] = stat.std
                row[f"{prefix}_NaN"] = stat.nan_pixels
                row[f"{prefix}_Inf"] = stat.inf_pixels
                row[f"{prefix}_ZeroPixels"] = stat.zero_pixels
                row[f"{prefix}_Empty"] = stat.empty_band

        return row

    def save_report(self) -> None:
        """
        Save validation results as CSV.
        """

        if not self.results:

            logger.warning("No validation results available.")
            return

        rows = [
            self._flatten_result(result)
            for result in self.results
        ]

        dataframe = pd.DataFrame(rows)

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            self.report_path,
            index=False,
        )

        logger.info(
            "Quality report written to %s",
            self.report_path,
        )

    # ======================================================
    # Dataset Processing
    # ======================================================

    def validate_dataset(self) -> None:
        """
        Validate every GeoTIFF inside datasets/raw.
        """

        scene_paths = sorted(
            [
                path
                for path in self.dataset_dir.iterdir()
                if path.is_file()
                and self._is_supported_file(path)
            ]
        )

        if not scene_paths:

            logger.warning(
                "No GeoTIFF files found in %s",
                self.dataset_dir,
            )
            return

        logger.info(
            "Found %d scenes.",
            len(scene_paths),
        )

        start_time = time.perf_counter()

        for scene in scene_paths:

            try:

                self.validate_scene(scene)

            except Exception:

                logger.exception(
                    "Validation failed for %s",
                    scene.name,
                )

        elapsed = time.perf_counter() - start_time

        self.save_report()

        passed = sum(
            result.valid
            for result in self.results
        )

        failed = len(self.results) - passed

        logger.info("=" * 70)
        logger.info("QUALITY VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info("Scenes Processed : %d", len(self.results))
        logger.info("Passed           : %d", passed)
        logger.info("Failed           : %d", failed)
        logger.info("Execution Time   : %.2f sec", elapsed)
        logger.info("=" * 70)


# ==========================================================
# Main
# ==========================================================


def main() -> None:
    """
    Entry point.
    """

    checker = QualityChecker()

    checker.validate_dataset()


if __name__ == "__main__":

    main()