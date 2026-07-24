"""Image tiling module."""

import numpy as np
from typing import List, Tuple


class ImageTiler:
    """Tile large images into smaller patches."""
    
    def __init__(self, tile_size: int = 256, overlap: int = 0):
        """Initialize tiler.
        
        Args:
            tile_size: Size of tiles
            overlap: Overlap between tiles
        """
        self.tile_size = tile_size
        self.overlap = overlap
    
    def tile(self, image: np.ndarray) -> List[np.ndarray]:
        """Tile image into patches.
        
        Args:
            image: Input image
            
        Returns:
            List of image tiles
        """
        tiles = []
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        
        for i in range(0, h - self.tile_size + 1, stride):
            for j in range(0, w - self.tile_size + 1, stride):
                tiles.append(image[i:i+self.tile_size, j:j+self.tile_size])
        
        return tiles