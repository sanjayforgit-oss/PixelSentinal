"""
Utility functions for PixelSentinel Evaluation.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation.logger import logger
from evaluation.config import config


def create_directory(directory: Path) -> None:
    """
    Creates a directory if it does not exist.
    """

    directory.mkdir(parents=True, exist_ok=True)


def create_output_directories() -> None:
    """
    Creates all output folders required by evaluation.
    """

    directories = [
        config.OUTPUT_DIR,
        config.LOG_DIR,
        config.COMPARISON_DIR,
        config.PLOTS_DIR,
    ]

    for directory in directories:
        create_directory(directory)


def list_images(folder: Path) -> List[Path]:
    """
    Returns all supported image files.
    """

    files = []

    for ext in config.IMAGE_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))

    return sorted(files)


def load_image(path: Path) -> np.ndarray:
    """
    Loads RGB image.
    """

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def save_csv(data: pd.DataFrame, output_path: Path) -> None:
    """
    Saves dataframe as CSV.
    """

    data.to_csv(output_path, index=False)

    logger.info("Saved CSV -> %s", output_path)


def pair_images(
    prediction_dir: Path,
    target_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    Matches predicted images with ground truth.
    """

    prediction_images = list_images(prediction_dir)

    pairs = []

    for pred in prediction_images:

        gt = target_dir / pred.name

        if gt.exists():
            pairs.append((pred, gt))
        else:
            logger.warning("Ground truth missing for %s", pred.name)

    return pairs


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Converts uint8 image to float32 [0,1]
    """

    image = image.astype(np.float32)

    image /= 255.0

    return image


def progress_bar(iterable):

    return tqdm(
        iterable,
        total=len(iterable),
        desc="Evaluating",
        unit="image",
    )