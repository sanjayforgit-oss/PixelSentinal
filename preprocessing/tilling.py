import os
import rasterio
import numpy as np
import cv2

def create_training_patches(scene_path="datasets/raw/scene001.tif", patch_size=256):
    """Slices the 7-band GeoTIFF into isolated inputs (4-band) and targets (3-band RGB) patches."""
    if not os.path.exists(scene_path):
        print(f"Error: Could not find {scene_path}. Make sure the download is finished!")
        return

    # Open the multi-band GeoTIFF
    with rasterio.open(scene_path) as src:
        # Read all 7 bands
        # Bands 1-4: SR_B5, SR_B6, SR_B7, ST_B10 (Inputs)
        # Bands 5-7: SR_B4, SR_B3, SR_B2 (Ground Truth Targets)
        data = src.read()
        
    channels, height, width = data.shape
    print(f"Loaded scene dimensions: {width}x{height} with {channels} bands.")

    # Setup directories for local PyTorch ingestion
    input_dir = "datasets/processed/train/input"
    target_dir = "datasets/processed/train/target"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    patch_count = 0

    # Slide across the image array grid
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            # Extract 4-channel Input Tensor (NIR, SWIR1, SWIR2, Thermal)
            input_patch = data[0:4, y:y+patch_size, x:x+patch_size]
            
            # Extract 3-channel Target Tensor (Red, Green, Blue)
            target_patch = data[4:7, y:y+patch_size, x:x+patch_size]
            
            # Basic Quality Control: Skip patches that are mostly blank or edge errors
            if np.max(input_patch) == 0 or np.max(target_patch) == 0:
                continue
                
            # Normalize to 0.0 - 1.0 baseline floating arrays for PyTorch training stability
            input_patch = input_patch.astype(np.float32) / 65535.0
            target_patch = target_patch.astype(np.float32) / 65535.0

            # Save Input patch as a NumPy binary file (since PNGs don't support 4 channels natively)
            np.save(os.path.join(input_dir, f"patch_{patch_count:04d}.npy"), input_patch)
            
            # Convert Target patch back to standard image form (H, W, C) and save as clean PNG
            # Rasterio reads as (C, H, W), OpenCV expects (H, W, C) in BGR order
            target_rgb = np.transpose(target_patch, (1, 2, 0))
            target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(target_dir, f"patch_{patch_count:04d}.png"), (target_bgr * 255).astype(np.uint8))
            
            patch_count += 1

    print(f"Successfully generated {patch_count} training pairs in 'datasets/processed/'!")

if __name__ == "__main__":
    create_training_patches()