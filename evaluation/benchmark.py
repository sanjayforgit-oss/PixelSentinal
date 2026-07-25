"""
PixelSentinel Benchmarking Framework

Evaluates batches of generated RGB satellite images and
computes benchmark statistics.

Author: Member 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import logging
import pandas as pd
from tqdm import tqdm

from .metrics import evaluate_metrics
from .utils import load_image


logger = logging.getLogger(__name__)


class Benchmark:
    """
    Benchmark generated RGB images against ground truth.
    """

    def __init__(
        self,
        prediction_dir: str | Path,
        target_dir: str | Path,
    ):

        self.prediction_dir = Path(prediction_dir)
        self.target_dir = Path(target_dir)

        if not self.prediction_dir.exists():
            raise FileNotFoundError(
                f"Prediction directory not found:\n{self.prediction_dir}"
            )

        if not self.target_dir.exists():
            raise FileNotFoundError(
                f"Target directory not found:\n{self.target_dir}"
            )

        self.results: List[Dict] = []

    # ------------------------------------------------------

    def image_pairs(self):

        """
        Match prediction images with ground-truth images
        using identical filenames.
        """

        prediction_files = sorted(
            self.prediction_dir.glob("*")
        )

        for pred_path in prediction_files:

            target_path = self.target_dir / pred_path.name

            if target_path.exists():

                yield pred_path, target_path

            else:

                logger.warning(
                    "Missing target image: %s",
                    pred_path.name,
                )

    # ------------------------------------------------------

    def evaluate(self):

        """
        Evaluate every image pair.
        """

        logger.info("Running benchmark...")

        self.results.clear()

        pairs = list(self.image_pairs())

        for pred_path, gt_path in tqdm(
            pairs,
            desc="Benchmark",
        ):

            pred = load_image(pred_path)
            gt = load_image(gt_path)

            metrics = evaluate_metrics(
                pred,
                gt,
            )

            metrics["image"] = pred_path.name

            self.results.append(metrics)

        logger.info(
            "Finished evaluating %d images.",
            len(self.results),
        )

            # ------------------------------------------------------

    def results_dataframe(self) -> pd.DataFrame:
        """
        Convert evaluation results into a pandas DataFrame.

        Returns:
            pd.DataFrame
        """

        if not self.results:
            return pd.DataFrame()

        columns = ["image"] + [
            c for c in self.results[0].keys()
            if c != "image"
        ]

        return pd.DataFrame(
            self.results,
            columns=columns,
        )

    # ------------------------------------------------------

    def average_metrics(self) -> Dict[str, float]:
        """
        Compute average value of every metric.

        Returns:
            Dictionary containing average metrics.
        """

        df = self.results_dataframe()

        if df.empty:
            return {}

        averages = {}

        for column in df.columns:

            if column == "image":
                continue

            averages[column] = float(df[column].mean())

        return averages

    # ------------------------------------------------------

    def best_images(
        self,
        metric: str = "PSNR",
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Return best-performing images.

        Metrics where higher is better:
            PSNR
            SSIM
            PCC
            UIQI

        Metrics where lower is better:
            MSE
            RMSE
            MAE
            SAM
            ERGAS
            LPIPS
        """

        df = self.results_dataframe()

        if df.empty:
            return df

        higher_is_better = {
            "PSNR",
            "SSIM",
            "PCC",
            "UIQI",
        }

        ascending = metric not in higher_is_better

        return (
            df.sort_values(
                metric,
                ascending=ascending,
            )
            .head(top_k)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------

    def worst_images(
        self,
        metric: str = "PSNR",
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Return worst-performing images.
        """

        df = self.results_dataframe()

        if df.empty:
            return df

        higher_is_better = {
            "PSNR",
            "SSIM",
            "PCC",
            "UIQI",
        }

        ascending = metric in higher_is_better

        return (
            df.sort_values(
                metric,
                ascending=ascending,
            )
            .head(top_k)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------

    def save_csv(
        self,
        output_path: str | Path,
    ) -> None:
        """
        Save image-wise benchmark results.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df = self.results_dataframe()

        df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Benchmark saved to %s",
            output_path,
        )

    # ------------------------------------------------------

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Return summary statistics.
        """

        averages = self.average_metrics()

        if not averages:
            return pd.DataFrame()

        return pd.DataFrame(
            [averages]
        )

    # ------------------------------------------------------

    def save_summary(
        self,
        output_path: str | Path,
    ) -> None:
        """
        Save average metrics.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = self.summary_dataframe()

        summary.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Summary saved to %s",
            output_path,
        )
