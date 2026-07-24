"""
PixelSentinel Dataset Builder
"""

import ee
import csv
from preprocessing.config import (
    GEE_PROJECT,
    LANDSAT_COLLECTIONS,
    YEARS,
    SEASONS,
    MAX_CLOUD_COVER,
    MAX_SCENES_PER_SEASON,
    INPUT_BANDS,
    TARGET_BANDS,
    ROI_LIST,
    METADATA_FILE,
    EXPORT_SCALE,
    EXPORT_FORMAT,
)

from preprocessing.logger import get_logger

logger = get_logger(__name__)

# -----------------------------
# Initialize Google Earth Engine
# -----------------------------
def initialize_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        logger.info("Successfully connected to Google Earth Engine.")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise

# -----------------------------
# Save Metadata
# -----------------------------
def save_metadata(
    filename: str,
    satellite: str,
    city: str,
    year: int,
    season: str,
    scene_id: str,
    acquisition_date: str,
    cloud_cover: float,
    export_status: str,
) -> None:
    """
    Save metadata for every exported scene.

    Parameters
    ----------
    filename : str
        Exported filename.

    satellite : str
        Landsat collection (Landsat 8 / Landsat 9).

    city : str
        ROI name.

    year : int
        Acquisition year.

    season : str
        Winter, Summer, Monsoon, etc.

    scene_id : str
        Earth Engine scene ID.

    acquisition_date : str
        Image acquisition date.

    cloud_cover : float
        Cloud cover percentage.

    export_status : str
        Started / Completed / Failed.
    """

    header = [
        "Filename",
        "Satellite",
        "City",
        "Year",
        "Season",
        "Scene_ID",
        "Acquisition_Date",
        "Cloud_Cover",
        "Export_Status",
    ]

    file_exists = METADATA_FILE.exists()

    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            filename,
            satellite,
            city,
            year,
            season,
            scene_id,
            acquisition_date,
            cloud_cover,
            export_status,
        ])

    logger.info(f"Metadata saved for {filename}")


# -----------------------------
# Create ROI
# -----------------------------
def create_roi(bounds):
    """
    Convert coordinates into a GEE Rectangle.
    """
    return ee.Geometry.Rectangle(bounds)
def fetch_collection(collection_name,roi, start_date, end_date):
    """
    Fetch a filtered Landsat 9 image collection for a given ROI.
    """

    logger.info(f"Querying {collection_name} archive...")

    collection = (
        ee.ImageCollection(collection_name)
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
        .sort("CLOUD_COVER")
    )

    count = collection.size().getInfo()

    logger.info(f"Found {count} candidate scenes.")

    return collection

def export_scene(image, roi, city_name, scene_index,satellite_name,year,season):
    """
    Create an Earth Engine export task for one Landsat scene.
    """

    
    export_name = f"{satellite_name}_{city_name}_{year}_{season}_{scene_index:03d}"
    logger.info(f"Creating export task: {export_name}")

    selected_image = image.select(INPUT_BANDS + TARGET_BANDS)

    task = ee.batch.Export.image.toDrive(
        image=selected_image,
        description=export_name,
        folder="PixelSentinel_Raw",
        fileNamePrefix=export_name,
        region=roi,
        scale=EXPORT_SCALE,
        maxPixels=1e13,
        fileFormat=EXPORT_FORMAT
    )

    try:
        task.start()
        status = task.status()

        logger.info(f"Task State : {status.get('state')}")
        logger.info(f"Task ID    : {status.get('id')}")
        logger.info(f"Started {export_name}")

    except Exception as e:
        logger.error(f"Failed {export_name}: {e}")
        return None
    properties = image.toDictionary([
        "CLOUD_COVER",
        "system:index"
    ]).getInfo()
    scene_id = properties["system:index"]
    cloud_cover = properties["CLOUD_COVER"]

    acquisition_date = (
        image.date()
        .format("YYYY-MM-dd")
        .getInfo()
    )

    

    metadata = {
        "filename": export_name,
        "satellite": satellite_name,
        "city": city_name,
        "year": year,
        "season": season,
        "scene_id": scene_id,
        "acquisition_date": acquisition_date,
        "cloud_cover": cloud_cover,
        "export_status": "Started",
    }

    save_metadata(**metadata)
    logger.info(f"Export task started: {export_name}")

    return task


# -----------------------------
# Build Dataset
# -----------------------------
def build_dataset(roi_list=None, years = None):
    """
    Build the PixelSentinel dataset by exporting Landsat imagery
    from multiple satellites, regions, years, and seasons.
    """
    if roi_list is None:
        roi_list = ROI_LIST
    if years is None:
        years = YEARS

    initialize_gee()
    logger.info("=" * 80)
    logger.info("Starting Dataset Collection")
    logger.info(f"ROIs  : {len(roi_list)}")
    logger.info(f"Years : {years}")
    logger.info("=" * 80)
    total_export_tasks = 0

    # Loop through Landsat 8 and Landsat 9
    for collection_name in LANDSAT_COLLECTIONS:

        satellite_name = (
            "Landsat8"
            if "LC08" in collection_name
            else "Landsat9"
        )

        logger.info("=" * 80)
        logger.info(f"Using {satellite_name}")

        # Loop through all ROIs
        for city_name, bounds in roi_list.items():

            logger.info("-" * 80)
            logger.info(f"Processing ROI: {city_name}")

            roi = create_roi(bounds)

            # Loop through years
            for year in years:

                # Loop through seasons
                for season_name, (start_suffix, end_suffix) in SEASONS.items():

                    # --------------------------------------------------
                    # Dynamic date generation
                    # --------------------------------------------------
                    start_date = f"{year}{start_suffix}"
                    end_date = f"{year}{end_suffix}"

                    logger.info(
                        f"{city_name} | {year} | {season_name}"
                    )

                    collection = fetch_collection(
                        collection_name=collection_name,
                        roi=roi,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    total_images = collection.size().getInfo()

                    if total_images == 0:
                        logger.warning(
                            f"No images found "
                            f"({city_name}, {year}, {season_name})"
                        )
                        continue

                    logger.info(
                        f"Found {total_images} candidate scenes."
                    )

                    images = collection.toList(total_images)

                    export_limit = min(
                        MAX_SCENES_PER_SEASON,
                        total_images,
                    )

                    for scene_index in range(export_limit):

                        image = ee.Image(images.get(scene_index))

                        export_scene(
                            image=image,
                            roi=roi,
                            city_name=city_name,
                            satellite_name=satellite_name,
                            year=year,
                            season=season_name,
                            scene_index=scene_index + 1,
                        )

                        # ------------------------------------------
                        # Metadata
                        # ------------------------------------------
                        '''
                        save_metadata(
                            filename=f"{satellite_name}_{city_name}_{year}_{season_name}_{scene_index+1:03d}",
                            satellite=satellite_name,
                            city=city_name,
                            year=year,
                            season=season_name,
                            scene_id=image.id().getInfo(),
                            acquisition_date=image.date().format(
                                "YYYY-MM-dd"
                            ).getInfo(),
                            cloud_cover=image.get(
                                "CLOUD_COVER"
                            ).getInfo(),
                            export_status="Started",
                        )
                        ''' # calling save_metadata inside export_scene()
                        total_export_tasks += 1

    logger.info("=" * 80)
    logger.info(
        f"Dataset generation complete."
    )
    logger.info(
        f"Total export tasks created: {total_export_tasks}"
    )

if __name__ == "__main__":
    build_dataset()