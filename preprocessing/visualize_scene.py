import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def visualize_landsat_channels(scene_path="datasets/raw/scene001.tif"):
    if not os.path.exists(scene_path):
        print(f"Error: Could not find {scene_path}. Make sure it is in your datasets/raw/ folder!")
        return

    # Open the multi-band image
    with rasterio.open(scene_path) as src:
        data = src.read()
        
    print(f"Image structural format: {data.shape} (Bands, Height, Width)")
    
    # Extract individual layers from the array stack
    # Inputs (Normalized just for display visualization purposes)
    nir = data[0].astype(np.float32)
    swir1 = data[1].astype(np.float32)
    thermal = data[3].astype(np.float32)
    
    # Ground Truth Targets (Bands 5, 6, 7 are R, G, B channels)
    r = data[4].astype(np.float32)
    g = data[5].astype(np.float32)
    b = data[6].astype(np.float32)
    
    # Stacking individual R, G, B channels into a single preview matrix
    rgb_stack = np.dstack((r, g, b))
    
    # Contrast stretching helper to map 16-bit satellite integers cleanly to 0-1 display floats
    def stretch(band):
        p2, p98 = np.percentile(band, (2, 98))
        if p98 - p2 == 0:
            return np.zeros_like(band)
        return np.clip((band - p2) / (p98 - p2), 0, 1)

    # Apply look-up stretching to make features human-readable
    rgb_stretched = np.dstack((stretch(r), stretch(g), stretch(b)))
    nir_stretched = stretch(nir)
    swir_stretched = stretch(swir1)
    thermal_stretched = stretch(thermal)

    # Plot the grid configuration layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(rgb_stretched)
    axes[0, 0].set_title("Ground Truth target: True Color (RGB)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(nir_stretched, cmap='gray')
    axes[0, 1].set_title("Input Channel 1: Near-Infrared (NIR - SR_B5)")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(swir_stretched, cmap='copper')
    axes[1, 0].set_title("Input Channel 2: Shortwave Infrared (SWIR 1 - SR_B6)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(thermal_stretched, cmap='magma')
    axes[1, 1].set_title("Input Channel 4: Thermal Infrared (TIRS - ST_B10)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_landsat_channels()