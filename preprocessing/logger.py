"""
===============================================================================
PixelSentinel - Logging Utility
-------------------------------------------------------------------------------
Author      : PixelSentinel Team
Project     : ISRO Bharatiya Antariksh Hackathon 2026

Purpose:
    Provides a centralized logging system for all preprocessing modules.

Features:
    - Console logging
    - File logging
    - Timestamped logs
    - Colored log levels (optional in terminal)
===============================================================================
"""

import logging
from pathlib import Path

from preprocessing.config import LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Create or return a configured logger.

    Parameters
    ----------
    name : str
        Name of the module requesting the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # ---------------------------------------------------------
    # Formatter
    # ---------------------------------------------------------

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------------------------------------------------------
    # Console Handler
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # File Handler
    # ---------------------------------------------------------

    log_path = Path(LOG_FILE)

    file_handler = logging.FileHandler(
        filename=log_path,
        mode="a",
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # Register Handlers
    # ---------------------------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


if __name__ == "__main__":

    logger = get_logger("LoggerTest")

    logger.info("Logger initialized successfully.")

    logger.warning("This is a warning.")

    logger.error("This is an error.")

    logger.debug("Debug message.")