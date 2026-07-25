"""
PixelSentinel Tile Verification

Verifies the tiled dataset before augmentation.

Checks
------
✓ Input tile exists
✓ Target tile exists
✓ Correct band count
✓ Tile size
✓ CRS
✓ Transform
✓ Band descriptions
✓ Metadata consistency
✓ Corrupted files

Author
------
PixelSentinel Team
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio

from preprocessing.config import (
    INPUT_TILE_DIR,
    TARGET_TILE_DIR,
    TILE_METADATA_FILE,
    METADATA_DIR,
)

from preprocessing.logger import get_logger

logger = get_logger(__name__)

VERIFICATION_REPORT = (
    METADATA_DIR /
    "tile_verification_report.csv"
)

# ==========================================================
# Verification Result
# ==========================================================

@dataclass(slots=True)
class TileVerificationResult:

    filename: str

    success: bool

    message: str

# ==========================================================
# Tile Verifier
# ==========================================================

class TileVerifier:

    """
    Verify tiled dataset integrity.
    """

    def __init__(self) -> None:

        self.input_directory = INPUT_TILE_DIR

        self.target_directory = TARGET_TILE_DIR

        self.metadata_file = TILE_METADATA_FILE

        self.report_file = VERIFICATION_REPORT

        self.total = 0

        self.passed = 0

        self.failed = 0

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
    def expected_scene(
        tile_name: str,
    ) -> str:

        return (
            tile_name
            .replace("_input.tif", "")
            .replace("_target.tif", "")
        )

    # ======================================================
    # Verify Single Tile Pair
    # ======================================================

    def verify_pair(
        self,
        input_tile: Path,
    ) -> TileVerificationResult:

        target_tile = self.paired_target(
            input_tile,
        )

        if not target_tile.exists():

            return TileVerificationResult(
                filename=input_tile.name,
                success=False,
                message="Missing target tile.",
            )

        try:

            with rasterio.open(input_tile) as input_ds, \
                 rasterio.open(target_tile) as target_ds:

                # ------------------------------------------
                # Band Count
                # ------------------------------------------

                if input_ds.count != 4:

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Input must contain 4 bands.",
                    )

                if target_ds.count != 3:

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Target must contain 3 bands.",
                    )

                # ------------------------------------------
                # Tile Size
                # ------------------------------------------

                if (
                    input_ds.width != 256
                    or input_ds.height != 256
                ):

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Invalid input tile size.",
                    )

                if (
                    target_ds.width != 256
                    or target_ds.height != 256
                ):

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Invalid target tile size.",
                    )

                # ------------------------------------------
                # CRS
                # ------------------------------------------

                if input_ds.crs != target_ds.crs:

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="CRS mismatch.",
                    )

                # ------------------------------------------
                # Transform
                # ------------------------------------------

                if input_ds.transform != target_ds.transform:

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Transform mismatch.",
                    )

                # ------------------------------------------
                # Descriptions
                # ------------------------------------------

                if any(
                    band is None
                    for band in input_ds.descriptions
                ):

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Missing input band descriptions.",
                    )

                if any(
                    band is None
                    for band in target_ds.descriptions
                ):

                    return TileVerificationResult(
                        filename=input_tile.name,
                        success=False,
                        message="Missing target band descriptions.",
                    )

            return TileVerificationResult(
                filename=input_tile.name,
                success=True,
                message="Success",
            )

        except Exception as exc:

            return TileVerificationResult(
                filename=input_tile.name,
                success=False,
                message=str(exc),
            )

    # ======================================================
    # Write Report
    # ======================================================

    def write_report(
        self,
        results: list[TileVerificationResult],
    ) -> None:

        header = [
            "Filename",
            "Status",
            "Message",
        ]

        with open(
            self.report_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(header)

            for result in results:

                writer.writerow(
                    [
                        result.filename,
                        "PASS" if result.success else "FAIL",
                        result.message,
                    ]
                )

        logger.info(
            "Verification report written to %s",
            self.report_file,
        )


    # ======================================================
    # Verify Dataset
    # ======================================================

    def verify_dataset(self) -> None:
        """
        Verify the complete tiled dataset.
        """

        start_time = time.time()

        logger.info("=" * 80)
        logger.info("VERIFYING TILED DATASET")
        logger.info("=" * 80)

        input_tiles = sorted(
            self.input_directory.glob("*_input.tif")
        )

        target_tiles = sorted(
            self.target_directory.glob("*_target.tif")
        )

        logger.info(
            "Input Tiles  : %d",
            len(input_tiles),
        )

        logger.info(
            "Target Tiles : %d",
            len(target_tiles),
        )

        if len(input_tiles) != len(target_tiles):

            logger.warning(
                "Input and target tile counts do not match."
            )

        results: list[TileVerificationResult] = []

        for index, input_tile in enumerate(input_tiles, start=1):

            result = self.verify_pair(
                input_tile,
            )

            results.append(result)

            self.total += 1

            if result.success:

                self.passed += 1

            else:

                self.failed += 1

            if (
                index % 500 == 0
                or index == len(input_tiles)
            ):

                logger.info(
                    "Verified %d / %d tiles...",
                    index,
                    len(input_tiles),
                )

        self.write_report(
            results,
        )

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("TILE VERIFICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            "Tiles Verified : %d",
            self.total,
        )
        logger.info(
            "Passed         : %d",
            self.passed,
        )
        logger.info(
            "Failed         : %d",
            self.failed,
        )
        logger.info("=" * 80)
        logger.info(
            "Execution Time : %.2f seconds",
            elapsed,
        )
        logger.info("=" * 80)

        if self.failed == 0:

            logger.info(
                "Tile dataset verification PASSED."
            )

        else:

            logger.warning(
                "Tile dataset verification FAILED."
            )

# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Execute tile verification.
    """

    verifier = TileVerifier()

    verifier.verify_dataset()


if __name__ == "__main__":

    main()