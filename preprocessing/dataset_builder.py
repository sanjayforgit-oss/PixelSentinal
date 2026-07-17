"""
===============================================================================
PixelSentinel Dataset Builder
-------------------------------------------------------------------------------

Author      : PixelSentinel Team

Project     : ISRO Bharatiya Antariksh Hackathon 2026

Purpose
-------
Collects high-quality multi-season Landsat imagery for training
PixelSentinel.

Pipeline

Initialize Earth Engine

↓

Load Landsat 8 + Landsat 9

↓

Loop through Regions

↓

Loop through Seasons

↓

Loop through Years

↓

Find Best Scene

↓

Validate Scene

↓

Export to Google Drive

↓

Save Metadata

===============================================================================
"""

from pathlib import Path
from typing import Optional

import ee

from preprocessing.config import *

from preprocessing.logger import get_logger

from preprocessing.utils import (
    initialize_gee,
    get_landsat_collection,
    build_date_range,
    create_region_directory,
    create_season_directory,
    generate_scene_filename,
)

from preprocessing.metadata import MetadataManager

from preprocessing.quality import SceneQuality

logger = get_logger(__name__)

class DatasetBuilder:
    """
    Main Dataset Builder.

    Responsible for creating the complete PixelSentinel dataset.
    """

    def __init__(self):

        logger.info("=" * 80)
        logger.info("Initializing Dataset Builder")
        logger.info("=" * 80)

        initialize_gee()

        self.collection = get_landsat_collection()

        self.metadata = MetadataManager()

        self.total_exports = 0

        self.failed_exports = 0

        logger.info("Dataset Builder Ready.")
    

    def load_collection(
        self,
        roi,
        start_date: str,
        end_date: str
    ) -> ee.ImageCollection:
        """
        Load merged Landsat collection.

        Parameters
        ----------
        roi

        start_date

        end_date

        Returns
        -------
        ee.ImageCollection
        """

        logger.info(
            f"Searching imagery "
            f"{start_date} -> {end_date}"
        )

        collection = (

            self.collection

            .filterBounds(roi)

            .filterDate(start_date, end_date)

            .filter(
                ee.Filter.lt(
                    "CLOUD_COVER",
                    MAX_CLOUD_COVER
                )
            )

            .sort("CLOUD_COVER")

        )

        return collection
    
    def find_best_scene(
        self,
        collection: ee.ImageCollection
    ) -> Optional[ee.Image]:

        """
        Returns lowest-cloud scene.
        """

        try:

            image = ee.Image(collection.first())

            if SceneQuality.validate(image):

                return image

            return None

        except Exception as e:

            logger.error(e)

            return None
    

    def prepare_image(
        self,
        image: ee.Image
    ) -> ee.Image:
        """
        Select PixelSentinel bands.

        Inputs

        B5
        B6
        B7
        B10

        Targets

        B4
        B3
        B2
        """

        logger.info("Selecting required bands.")

        image = image.select(

            [

                "SR_B5",

                "SR_B6",

                "SR_B7",

                "ST_B10",

                "SR_B4",

                "SR_B3",

                "SR_B2"

            ]

        )

        return image
    

    def process_year(

        self,

        roi,

        region_name: str,

        season_name: str,

        year: int,

        season_dates

    ):

        """
        Process one season of one year.
        """

        start_date, end_date = build_date_range(

            year,

            season_dates

        )

        logger.info(

            f"{region_name}"

            f" | "

            f"{season_name}"

            f" | "

            f"{year}"

        )

        collection = self.load_collection(

            roi,

            start_date,

            end_date

        )

        image = self.find_best_scene(

            collection

        )

        if image is None:

            logger.warning(

                "No suitable scene found."

            )

            return

        image = self.prepare_image(image)

        logger.info(

            f"Best Scene: "

            f"{SceneQuality.scene_id(image)}"

        )

        #
        # Export comes in Part 2
        #
        self.queue_export(

            image=image,

            roi=roi,

            region_name=region_name,

            season_name=season_name,

            year=year

        )


    def process_season(

        self,

        roi,

        region_name,

        season_name,

        season_dates

    ):

        """
        Process all configured years.
        """

        logger.info(

            "=" * 60

        )

        logger.info(

            f"Season: {season_name}"

        )

        logger.info(

            "=" * 60

        )

        for year in YEARS:

            self.process_year(

                roi,

                region_name,

                season_name,

                year,

                season_dates

            )

    def process_region(

        self,

        region_name,

        roi

    ):

        """
        Process one geographical region.
        """

        logger.info(

            "#" * 80

        )

        logger.info(

            f"Region : {region_name}"

        )

        logger.info(

            "#" * 80

        )

        create_region_directory(

            RAW_DATA_DIR,

            region_name

        )

        for season_name, season_dates in SEASONS.items():

            create_season_directory(

                RAW_DATA_DIR / region_name,

                season_name

            )

            self.process_season(

                roi,

                region_name,

                season_name,

                season_dates

            )


###############################################################################
# Export Utilities
###############################################################################

    def build_export_name(
        self,
        region_name: str,
        season_name: str,
        year: int,
        image: ee.Image
    ) -> str:
        """
        Generate a unique export filename.
        """

        scene_id = SceneQuality.scene_id(image)

        filename = (
            f"{region_name}_"
            f"{season_name}_"
            f"{year}_"
            f"{scene_id}"
        )

        filename = filename.replace("/", "_")

        return filename
    
    def export_scene(
        self,
        image: ee.Image,
        roi,
        export_name: str
    ):

        """
        Create Google Earth Engine Export Task.
        """

        logger.info(
            f"Creating Export Task : {export_name}"
        )

        task = ee.batch.Export.image.toDrive(

            image=image,

            description=export_name,

            folder=GDRIVE_EXPORT_FOLDER,

            fileNamePrefix=export_name,

            region=roi,

            scale=EXPORT_SCALE,

            crs=EXPORT_CRS,

            fileFormat="GeoTIFF",

            maxPixels=1e13

        )

        task.start()

        logger.info(
            f"Export Started : {export_name}"
        )

        self.total_exports += 1

        return task
    
    def save_scene_metadata(
        self,
        image: ee.Image,
        export_name: str,
        region_name: str,
        season_name: str,
        year: int
    ):

        """
        Save metadata for each exported scene.
        """

        metadata = {

            "filename": export_name,

            "scene_id": SceneQuality.scene_id(image),

            "satellite": SceneQuality.satellite(image),

            "date": SceneQuality.acquisition_date(image),

            "cloud_cover": SceneQuality.cloud_cover(image),

            "region": region_name,

            "season": season_name,

            "year": year

        }

        self.metadata.add_scene(metadata)

        logger.info(
            "Metadata Saved."
        )


    def scene_already_processed(
        self,
        export_name: str
    ) -> bool:

        """
        Skip duplicate exports.
        """

        return self.metadata.scene_exists(export_name)
    

    def queue_export(
        self,
        image: ee.Image,
        roi,
        region_name: str,
        season_name: str,
        year: int
    ):

        export_name = self.build_export_name(

            region_name,

            season_name,

            year,

            image

        )

        if self.scene_already_processed(export_name):

            logger.info(
                f"Skipping existing scene : {export_name}"
            )

            return

        self.export_scene(

            image,

            roi,

            export_name

        )

        self.save_scene_metadata(

            image,

            export_name,

            region_name,

            season_name,

            year

        )



###############################################################################
# Summary
###############################################################################

    def print_summary(self):

        logger.info("")
        logger.info("=" * 80)
        logger.info("PIXELSENTINEL DATASET SUMMARY")
        logger.info("=" * 80)

        logger.info(
            f"Export Tasks Created : {self.total_exports}"
        )

        logger.info(
            f"Failed Regions       : {self.failed_exports}"
        )

        logger.info(
            f"Metadata Records     : {self.metadata.total_records()}"
        )

        logger.info("=" * 80)

###############################################################################
# Build Complete Dataset
###############################################################################

    def build_dataset(self):
        """
        Main orchestration function.

        Iterates through every configured region,
        every season,
        every year,
        and creates export tasks.
        """

        logger.info("")
        logger.info("=" * 80)
        logger.info("STARTING PIXELSENTINEL DATASET GENERATION")
        logger.info("=" * 80)

        logger.info(f"Regions : {len(REGIONS)}")
        logger.info(f"Years   : {len(YEARS)}")
        logger.info(f"Seasons : {len(SEASONS)}")

        estimated_exports = (
            len(REGIONS)
            * len(YEARS)
            * len(SEASONS)
        )

        logger.info(
            f"Estimated Export Tasks : {estimated_exports}"
        )

        logger.info("=" * 80)

        for region_name, roi in REGIONS.items():

            try:

                self.process_region(
                    region_name,
                    roi
                )

            except Exception as e:

                logger.error(
                    f"Region failed : {region_name}"
                )

                logger.error(e)

                continue

        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPORT TASK CREATION FINISHED")
        logger.info("=" * 80)


###############################################################################
# Task Monitoring
###############################################################################

    def monitor_tasks(self):
        """
        Prints status of all active
        Earth Engine export tasks.
        """

        logger.info("")
        logger.info("=" * 80)
        logger.info("CURRENT EARTH ENGINE TASK STATUS")
        logger.info("=" * 80)

        tasks = ee.batch.Task.list()

        if len(tasks) == 0:

            logger.warning(
                "No Earth Engine Tasks Found."
            )

            return

        for task in tasks:

            status = task.status()

            logger.info(

                f"{status['description']}"

                f" | "

                f"{status['state']}"

            )

        logger.info("=" * 80)


###############################################################################
# Main Entry
###############################################################################

def main():

    builder = DatasetBuilder()

    builder.build_dataset()

    builder.monitor_tasks()

    builder.print_summary()


if __name__ == "__main__":

    main()