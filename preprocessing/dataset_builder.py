import os
import ee
import rasterio

def initialize_gee(project_name="pixelsentinel"):
    """Initializes Google Earth Engine with the specific project id."""
    try:
        ee.Initialize(project=project_name)
        print(f"Successfully connected to Earth Engine project: {project_name}")
    except Exception as e:
        print(f"Initialization failed. Make sure you ran 'earthengine authenticate'. Error: {e}")

def build_training_dataset():
    # Define an area of interest over India (Approx regional box)
    # Format: ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    #roi = ee.Geometry.Rectangle([79.8, 12.8, 80.4, 13.2])
    # Tighter bounding box over the region to stay under the 48MB direct download cap
    roi = ee.Geometry.Rectangle([79.95, 12.95, 80.25, 13.15])
    
    print("Querying Landsat 9 archive...")
    # Filter for low cloud-cover imagery from Landsat 9 Collection 2 Level-2
    collection = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                  .filterBounds(roi)
                  .filterDate("2025-01-01", "2026-06-01")
                  .filter(ee.Filter.lt("CLOUD_COVER", 5))
                  .sort("CLOUD_COVER"))
    
    # Grab the clearest image match
    image = ee.Image(collection.first())
    
    if image is None:
        print("No low-cloud imagery found for the specified criteria.")
        return
        
    print(f"Found clear scene ID: {image.id().getInfo()}")
    
    # Isolate our specific PixelSentinel 7 bands
    # Inputs: B5 (NIR), B6 (SWIR 1), B7 (SWIR 2), B10 (Thermal)
    # Ground Truth Targets: B4 (Red), B3 (Green), B2 (Blue)
    selected_bands = image.select([
        'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 
        'SR_B4', 'SR_B3', 'SR_B2'
    ])
    
    # Create target data directories if they don't exist
    os.makedirs("datasets/raw", exist_ok=True)
    
    # Get download URL for the pre-filtered, lightweight multi-band image
    download_url = selected_bands.getDownloadURL({
        'scale': 30,
        'crs': image.projection(),
        'region': roi,
        'format': 'GEO_TIFF'
    })
    
    print(f"\nSuccessfully generated lightweight cloud pipeline!")
    print(f"Download URL link: {download_url}")
    print("\nCopy this link into your browser or use curl/wget to download the file directly into your 'datasets/raw/' directory as 'scene001.tif'.")

if __name__ == "__main__":
    initialize_gee()
    build_training_dataset()