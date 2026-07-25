"""
Evaluation configuration for PixelSentinel.

Author: Member 3
Project: PixelSentinel
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationConfig:
    """
    Stores all configuration required for the evaluation module.
    """

    # Root directory (PixelSentinel/)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # Dataset Paths
    TEST_INPUT_DIR: Path = PROJECT_ROOT / "datasets" / "test" / "inputs"
    TEST_TARGET_DIR: Path = PROJECT_ROOT / "datasets" / "test" / "targets"
    PREDICTION_DIR: Path = PROJECT_ROOT / "datasets" / "test" / "predictions"

    # Output Directory
    OUTPUT_DIR: Path = PROJECT_ROOT / "evaluation_results"

    METRICS_CSV: Path = OUTPUT_DIR / "metrics.csv"
    BENCHMARK_CSV: Path = OUTPUT_DIR / "benchmark.csv"
    SUMMARY_CSV: Path = OUTPUT_DIR / "summary.csv"

    REPORT_PDF: Path = OUTPUT_DIR / "report.pdf"

    LOG_DIR: Path = OUTPUT_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "evaluation.log"

    COMPARISON_DIR: Path = OUTPUT_DIR / "comparisons"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"

    # Supported image extensions
    IMAGE_EXTENSIONS: List[str] = field(
        default_factory=lambda: [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
        ]
    )

    # Image settings
    IMAGE_SIZE = (256, 256)

    # Pixel Range
    DATA_RANGE = 255

    # Metrics
    ENABLE_PSNR = True
    ENABLE_SSIM = True
    ENABLE_RMSE = True
    ENABLE_MAE = True
    ENABLE_SAM = True
    ENABLE_LPIPS = True

    # Plot settings
    DPI = 300

    FIGURE_SIZE = (8, 6)

    # Logging
    LOG_LEVEL = "INFO"


config = EvaluationConfig()
