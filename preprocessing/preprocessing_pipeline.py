"""
PixelSentinel — Preprocessing Pipeline Entry Point (V2.0)

Main Command-Line Interface (CLI) for running the automated end-to-end 
dataset preprocessing framework.

Usage:
    python -m preprocessing.preprocessing_pipeline
    python -m preprocessing.preprocessing_pipeline --no-clean --no-incoming
"""

import argparse
import logging
import sys
from pathlib import Path

# Import manager and configuration defaults
try:
    from preprocessing.config import (
        AUTO_CLEAN_OUTPUT,
        AUTO_COPY_INCOMING,
        LOG_LEVEL,
        PIPELINE_VERSION,
        REPORT_DIR,
        STOP_ON_ERROR,
    )
    from preprocessing.pipeline_manager import PipelineManager
except ImportError:
    # Fallback configuration if module resolution differs
    PIPELINE_VERSION = "2.0"
    AUTO_COPY_INCOMING = True
    AUTO_CLEAN_OUTPUT = True
    STOP_ON_ERROR = True
    LOG_LEVEL = "INFO"
    REPORT_DIR = Path("datasets/reports")
    PipelineManager = None


def setup_pipeline_logging(log_level: str = "INFO") -> None:
    """Configures console and file logging for the pipeline execution."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = REPORT_DIR / "pipeline_execution.log"

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger setup
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8")
        ]
    )


def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments for pipeline execution overrides."""
    parser = argparse.ArgumentParser(
        description=f"PixelSentinel Preprocessing Pipeline V{PIPELINE_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--no-incoming",
        action="store_true",
        help="Skip copying files from the incoming directory to the raw directory."
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean the active preprocessing workspace before running."
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Continue running remaining pipeline stages even if a stage fails"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity level"
    )

    return parser.parse_args()


def main() -> None:
    """Main execution workflow."""
    args = parse_cli_args()

    # Configure logging
    setup_pipeline_logging(args.log_level)
    logger = logging.getLogger("PixelSentinel.CLI")

    logger.info(f"Starting PixelSentinel Preprocessing Pipeline (V{PIPELINE_VERSION})")

    if PipelineManager is None:
        logger.critical("Failed to load PipelineManager. Check your project structure and config imports.")
        sys.exit(1)

    # Resolve settings from CLI arguments or global defaults
    copy_incoming = AUTO_COPY_INCOMING and (not args.no_incoming)

    clean_output = AUTO_CLEAN_OUTPUT and (not args.no_clean)

    stop_on_err = STOP_ON_ERROR and (not args.ignore_errors)

    # Initialize and execute pipeline manager
    manager = PipelineManager(
        auto_copy_incoming=copy_incoming,
        auto_clean_output=clean_output,
        stop_on_error=stop_on_err
    )

    success = manager.run()

    if success:
        logger.info("Pipeline executed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline finished with errors. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()