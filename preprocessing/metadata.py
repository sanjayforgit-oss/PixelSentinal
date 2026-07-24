"""
===============================================================================
PixelSentinel - Metadata Manager
-------------------------------------------------------------------------------
Author      : PixelSentinel Team
Project     : ISRO Bharatiya Antariksh Hackathon 2026

Purpose:
    Handles creation, storage and management of dataset metadata.

Metadata Stored
---------------
• Scene ID
• Satellite
• ROI
• Season
• Year
• Acquisition Date
• Cloud Cover
• Download Path
• Download Time

This file creates a CSV database that can later be used for:
    - Dataset statistics
    - Debugging
    - Experiment reproducibility
    - Training reports
===============================================================================
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

from preprocessing.config import (
    METADATA_DIR,
    METADATA_FILENAME,
)

from preprocessing.logger import get_logger
from preprocessing.utils import get_satellite_name

logger = get_logger(__name__)


class MetadataManager:
    """
    Handles metadata creation and storage.
    """

    def __init__(self):

        self.metadata_file = METADATA_DIR / METADATA_FILENAME

        self.columns = [
            "Scene_ID",
            "Satellite",
            "Region",
            "Season",
            "Year",
            "Acquisition_Date",
            "Cloud_Cover",
            "File_Path",
            "Download_Time"
        ]

        self._initialize_csv()

    # -----------------------------------------------------------------

    def _initialize_csv(self):
        """
        Create metadata CSV if it does not already exist.
        """

        if not self.metadata_file.exists():

            df = pd.DataFrame(columns=self.columns)

            df.to_csv(self.metadata_file, index=False)

            logger.info(
                f"Created metadata file: {self.metadata_file}"
            )

    # -----------------------------------------------------------------

    def add_scene(
        self,
        scene_id: str,
        region: str,
        season: str,
        year: int,
        acquisition_date: str,
        cloud_cover: float,
        file_path: Path,
    ):
        """
        Add a downloaded scene to metadata.
        """

        satellite = get_satellite_name(scene_id)

        record = {

            "Scene_ID": scene_id,

            "Satellite": satellite,

            "Region": region,

            "Season": season,

            "Year": year,

            "Acquisition_Date": acquisition_date,

            "Cloud_Cover": cloud_cover,

            "File_Path": str(file_path),

            "Download_Time":
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        df = pd.read_csv(self.metadata_file)

        df = pd.concat(
            [df, pd.DataFrame([record])],
            ignore_index=True
        )

        df.to_csv(self.metadata_file, index=False)

        logger.info(
            f"Metadata added for scene: {scene_id}"
        )

    # -----------------------------------------------------------------

    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata CSV.

        Returns
        -------
        pandas.DataFrame
        """

        return pd.read_csv(self.metadata_file)

    # -----------------------------------------------------------------

    def total_scenes(self) -> int:
        """
        Return total downloaded scenes.
        """

        df = self.load_metadata()

        return len(df)

    # -----------------------------------------------------------------

    def scenes_by_region(self):

        df = self.load_metadata()

        return df.groupby("Region").size()

    # -----------------------------------------------------------------

    def scenes_by_satellite(self):

        df = self.load_metadata()

        return df.groupby("Satellite").size()

    # -----------------------------------------------------------------

    def scenes_by_season(self):

        df = self.load_metadata()

        return df.groupby("Season").size()

    # -----------------------------------------------------------------

    def summary(self):
        """
        Print dataset summary.
        """

        df = self.load_metadata()

        logger.info("=" * 70)

        logger.info(f"Total Scenes : {len(df)}")

        logger.info("")

        logger.info("Scenes by Region")

        logger.info(df["Region"].value_counts())

        logger.info("")

        logger.info("Scenes by Satellite")

        logger.info(df["Satellite"].value_counts())

        logger.info("")

        logger.info("Scenes by Season")

        logger.info(df["Season"].value_counts())

        logger.info("=" * 70)


if __name__ == "__main__":

    manager = MetadataManager()

    print(manager.load_metadata())