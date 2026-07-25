"""
Logger for PixelSentinel Evaluation Module.
"""

import logging
from pathlib import Path

from evaluation.config import config


def setup_logger(name: str = "PixelSentinel") -> logging.Logger:
    """
    Creates and configures a reusable logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """

    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(config.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()