"""
Training scheduler for PixelSentinel.

This script repeatedly runs:
  1. `python -m colorization.train`
  2. checks the latest checkpoint epoch
  3. runs `validate_model.py` only when that epoch is divisible by 5
  4. waits for a configurable cooldown

It appends validation output to a CSV log in the project root.

Because this launcher does not modify existing files, validation is triggered
based on the checkpoint saved by the training run.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "latest_checkpoint.pth"
VALIDATION_CSV = ROOT / "validation_results.csv"


def run_command(command: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    return_code = process.wait()
    output = "".join(output_lines)
    if return_code != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n\nOUTPUT:\n{output}"
        )
    return output


def extract_validation_metrics(output: str) -> dict[str, float]:
    patterns = {
        "l1": r"L1 Loss\s*:\s*([0-9.]+)",
        "mse": r"MSE\s*:\s*([0-9.]+)",
        "psnr": r"PSNR \(dB\)\s*:\s*([0-9.]+)",
        "ssim": r"SSIM\s*:\s*([0-9.]+)",
        "accuracy": r"Accuracy\s*:\s*([0-9.]+)",
        "f1": r"F1 Score\s*:\s*([0-9.]+)",
        "lpips": r"LPIPS\s*:\s*([0-9.]+)",
        "delta_e": r"Delta E\s*:\s*([0-9.]+)",
    }

    metrics: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if not match:
            raise ValueError(f"Could not find '{key}' in validation output.")
        metrics[key] = float(match.group(1))
    return metrics


def append_validation_row(csv_path, rowdict):
    # 1. Include 'validation_run' in fieldnames
    fieldnames = [
        "iteration",
        "validation_run",
        "checkpoint_epoch",
        "timestamp",
        "l1",
        "mse",
        "psnr",
        "ssim",
        "accuracy",
        "f1",
        "lpips",
        "delta_e"
    ]
    
    file_exists = Path(csv_path).exists()
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        # 2. Add extrasaction='ignore' so it never crashes on extra keys
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        
        # Write header if file is newly created
        if not file_exists or Path(csv_path).stat().st_size == 0:
            writer.writeheader()
            
        writer.writerow(rowdict)


def get_last_completed_epoch() -> int:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        raise ValueError(f"Checkpoint {CHECKPOINT_PATH} does not contain a valid epoch number.")
    return epoch


def validate_and_log(iteration, validation_run, checkpoint_epoch):
    output = run_command([sys.executable, "validate_model.py"])
    print(output)
    # 1. Run validate_model.py and parse metrics output
    metrics = extract_validation_metrics(output)  # or regex extraction logic
    
    # 2. Build complete row dictionary
    row_data = {
        "iteration": iteration,
        "validation_run": validation_run,
        "checkpoint_epoch": checkpoint_epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metrics  # unpacked l1, mse, psnr, ssim, accuracy, f1, lpips, delta_e
    }
    
    # 3. Call with BOTH arguments (csv_path AND row_data)
    
    append_validation_row(VALIDATION_CSV, row_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training, cooldown, and conditional validation cycles.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="How many training cycles to run. Default: 2",
    )
    parser.add_argument(
        "--cooldown-minutes",
        type=int,
        default=30,
        help="Minutes to wait between training cycles. Default: 30",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=5,
        help="Run validation when the latest checkpoint epoch is divisible by this value. Default: 5",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.cooldown_minutes < 0:
        raise ValueError("--cooldown-minutes cannot be negative")
    if args.validation_every < 1:
        raise ValueError("--validation-every must be at least 1")

    for iteration in range(1, args.iterations + 1):
        print(f"\n=== Training cycle {iteration}/{args.iterations} ===")
        run_command([sys.executable, "-m", "colorization.train"])

        last_epoch = get_last_completed_epoch()
        print(f"\nLatest completed epoch: {last_epoch}")

        if last_epoch % args.validation_every == 0:
            print(f"\n=== Validation triggered at epoch {last_epoch} ===")
            validate_and_log(iteration=iteration, validation_run=1, checkpoint_epoch=last_epoch)
        else:
            print(
                f"\nSkipping validation because epoch {last_epoch} is not divisible by {args.validation_every}."
            )

        if iteration < args.iterations:
            wait_seconds = args.cooldown_minutes * 60
            print(f"\nWaiting for {args.cooldown_minutes} minute(s) before the next cycle...")
            time.sleep(wait_seconds)

    print(f"\nAll done. Validation log saved to: {VALIDATION_CSV}")


if __name__ == "__main__":
    main()
