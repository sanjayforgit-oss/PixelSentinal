"""
PixelSentinel Dataset Builder
"""

import ee

from preprocessing.config import (
    GEE_PROJECT,
    LANDSAT_COLLECTION,
    START_DATE,
    END_DATE,
    MAX_CLOUD_COVER,
    INPUT_BANDS,
    TARGET_BANDS,
    ROI_LIST,
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
# Create ROI
# -----------------------------
def create_roi(bounds):
    """
    Convert coordinates into a GEE Rectangle.
    """
    return ee.Geometry.Rectangle(bounds)
def fetch_collection(roi):
    """
    Fetch a filtered Landsat 9 image collection for a given ROI.
    """

    logger.info("Querying Landsat 9 archive...")

    collection = (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
        .sort("CLOUD_COVER")
    )

    count = collection.size().getInfo()

    logger.info(f"Found {count} candidate scenes.")

    return collection

def export_scene(image, roi, city_name, scene_index):
    """
    Create an Earth Engine export task for one Landsat scene.
    """

    export_name = f"{city_name}_scene_{scene_index:03d}"

    logger.info(f"Creating export task: {export_name}")

    selected_image = image.select(INPUT_BANDS + TARGET_BANDS)

    task = ee.batch.Export.image.toDrive(
        image=selected_image,
        description=export_name,
        folder="PixelSentinel_Raw",
        fileNamePrefix=export_name,
        region=roi,
        scale=30,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )

    task.start()

    logger.info(f"Export task started: {export_name}")

    return task
def build_dataset():
    """
    Build the PixelSentinel dataset by creating export tasks
    for each ROI defined in config.py.
    """

    initialize_gee()

    for city_name, bounds in ROI_LIST.items():

        logger.info("=" * 60)
        logger.info(f"Processing {city_name}")

        roi = create_roi(bounds)

        collection = fetch_collection(roi)

        total_images = collection.size().getInfo()

        if total_images == 0:
            logger.warning(f"No scenes found for {city_name}")
            continue

        logger.info(f"Found {total_images} scenes.")

        images = collection.toList(total_images)

        # Export the first 5 clearest scenes
        export_limit = min(5, total_images)

        for i in range(export_limit):

            image = ee.Image(images.get(i))

            export_scene(
                image=image,
                roi=roi,
                city_name=city_name,
                scene_index=i + 1
            )

        logger.info(f"Finished {city_name}")
if __name__ == "__main__":
    build_dataset()