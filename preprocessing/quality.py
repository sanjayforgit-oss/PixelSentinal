"""
===============================================================================
PixelSentinel - Scene Quality Assessment
-------------------------------------------------------------------------------
Author      : PixelSentinel Team
Project     : ISRO Bharatiya Antariksh Hackathon 2026

Purpose
-------
Performs automatic quality assessment of Landsat scenes before download.

Checks
------
1. Cloud Cover
2. Missing Data
3. Scene Validity
4. Band Availability
5. Download Readiness

===============================================================================
"""

from typing import Dict, Any

import ee

from preprocessing.config import MAX_CLOUD_COVER
from preprocessing.logger import get_logger

logger = get_logger(__name__)


class SceneQuality:

    """
    Utility class for evaluating Landsat scenes.
    """

    REQUIRED_BANDS = [
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
        "ST_B10"
    ]

    @staticmethod
    def scene_exists(image: ee.Image) -> bool:
        """
        Verify image is valid.
        """

        try:
            image.bandNames().getInfo()
            return True

        except Exception:

            logger.warning("Invalid Earth Engine image.")

            return False

    @staticmethod
    def has_required_bands(image: ee.Image) -> bool:
        """
        Check if all required bands exist.
        """

        try:

            bands = image.bandNames().getInfo()

            missing = []

            for band in SceneQuality.REQUIRED_BANDS:

                if band not in bands:

                    missing.append(band)

            if len(missing) > 0:

                logger.warning(
                    f"Missing bands: {missing}"
                )

                return False

            return True

        except Exception as e:

            logger.error(e)

            return False

    @staticmethod
    def cloud_cover(image: ee.Image) -> float:
        """
        Return cloud cover percentage.
        """

        try:

            return float(
                image.get("CLOUD_COVER").getInfo()
            )

        except Exception:

            return 100.0

    @staticmethod
    def passes_cloud_filter(image: ee.Image) -> bool:
        """
        Check cloud threshold.
        """

        cloud = SceneQuality.cloud_cover(image)

        if cloud <= MAX_CLOUD_COVER:

            return True

        logger.info(
            f"Rejected (Cloud Cover = {cloud:.2f}%)"
        )

        return False

    @staticmethod
    def acquisition_date(image: ee.Image) -> str:
        """
        Get acquisition date.
        """

        try:

            return ee.Date(
                image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        except Exception:

            return "Unknown"

    @staticmethod
    def scene_id(image: ee.Image) -> str:
        """
        Scene ID.
        """

        try:

            return image.id().getInfo()

        except Exception:

            return "Unknown"

    @staticmethod
    def satellite(image: ee.Image) -> str:
        """
        Landsat platform.
        """

        try:

            return image.get("SPACECRAFT_ID").getInfo()

        except Exception:

            return "Unknown"

    @staticmethod
    def image_size(image: ee.Image) -> Dict[str, Any]:
        """
        Returns image dimensions.
        """

        try:

            projection = image.select(0).projection()

            scale = projection.nominalScale().getInfo()

            return {

                "resolution": scale,

                "projection": projection.crs().getInfo()

            }

        except Exception:

            return {

                "resolution": None,

                "projection": None

            }

    @staticmethod
    def validate(image: ee.Image) -> bool:
        """
        Full validation pipeline.
        """

        logger.info("=" * 70)

        if not SceneQuality.scene_exists(image):

            logger.warning("Scene rejected.")

            return False

        if not SceneQuality.has_required_bands(image):

            logger.warning("Scene rejected.")

            return False

        if not SceneQuality.passes_cloud_filter(image):

            logger.warning("Scene rejected.")

            return False

        logger.info(
            f"Scene : {SceneQuality.scene_id(image)}"
        )

        logger.info(
            f"Satellite : {SceneQuality.satellite(image)}"
        )

        logger.info(
            f"Date : {SceneQuality.acquisition_date(image)}"
        )

        logger.info(
            f"Cloud : {SceneQuality.cloud_cover(image):.2f}%"
        )

        logger.info("=" * 70)

        return True