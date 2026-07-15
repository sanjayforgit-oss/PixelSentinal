"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_images(
    images: List[np.ndarray],
    titles: List[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """Plot multiple images.
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for idx, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image, cmap='gray')
        if titles:
            ax.set_title(titles[idx])
        ax.axis('off')
    plt.tight_layout()
    return fig