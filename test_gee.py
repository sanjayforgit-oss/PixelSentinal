from preprocessing.dataset_builder import (
    initialize_gee,
    create_roi,
    fetch_collection
)

from preprocessing.config import ROI_LIST

initialize_gee()

roi = create_roi(ROI_LIST["Chennai"])

collection = fetch_collection(roi)

print("Collection fetched successfully!")