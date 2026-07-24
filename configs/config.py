"""
PixelSentinel Configuration

This module centralizes all project configuration used by the
colorization pipeline.

Author:
    PixelSentinel Team

Project:
    ISRO Bharatiya Antariksh Hackathon 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


# ==============================================================================
# Project Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectConfig:
    """Project directory configuration."""

    root: Path = PROJECT_ROOT

    datasets: Path = field(default_factory=lambda: PROJECT_ROOT / "datasets")

    checkpoints: Path = field(
        default_factory=lambda: PROJECT_ROOT / "checkpoints"
    )

    docs: Path = field(default_factory=lambda: PROJECT_ROOT / "docs")

    notebooks: Path = field(
        default_factory=lambda: PROJECT_ROOT / "notebooks"
    )

    scripts: Path = field(
        default_factory=lambda: PROJECT_ROOT / "scripts"
    )


# ==============================================================================
# Dataset Configuration
# ==============================================================================


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""

    train_inputs: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "datasets"
        / "train"
        / "inputs"
    )

    train_targets: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "datasets"
        / "train"
        / "targets"
    )

    val_inputs: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "datasets"
        / "val"
        / "inputs"
    )

    val_targets: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "datasets"
        / "val"
        / "targets"
    )

    test_inputs: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "datasets"
        / "test"
        / "inputs"
    )

    test_targets: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "datasets"
        / "test"
        / "targets"
    )

    image_size: int = 256

    input_channels: int = 4

    output_channels: int = 3


# ==============================================================================
# Model Configuration
# ==============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Pix2Pix model configuration."""

    generator_features: int = 64

    discriminator_features: int = 64

    use_dropout: bool = True

    normalization: str = "batch"

    weight_initialization: str = "normal"


# ==============================================================================
# Training Configuration
# ==============================================================================


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 200

    batch_size: int = 8

    learning_rate: float = 2e-4

    beta1: float = 0.5

    beta2: float = 0.999

    lambda_l1: float = 100.0

    num_workers: int = 4

    shuffle: bool = True

    pin_memory: bool = True

    mixed_precision: bool = True

    save_every: int = 5

    random_seed: int = 42


# ==============================================================================
# Checkpoint Configuration
# ==============================================================================


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint configuration."""

    generator_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "checkpoints"
        / "generator"
    )

    discriminator_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "checkpoints"
        / "discriminator"
    )

    best_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "checkpoints"
        / "best"
    )

    resume_training: bool = False


# ==============================================================================
# Logging Configuration
# ==============================================================================


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    log_interval: int = 20

    verbose: bool = True

    tensorboard: bool = False


# ==============================================================================
# Device Configuration
# ==============================================================================


@dataclass(frozen=True)
class DeviceConfig:
    """Device configuration."""

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    use_cuda: bool = torch.cuda.is_available()

    gpu_count: int = torch.cuda.device_count()

    amp: bool = torch.cuda.is_available()


# ==============================================================================
# Master Configuration
# ==============================================================================


@dataclass(frozen=True)
class Config:
    """Master configuration."""

    project: ProjectConfig = field(default_factory=ProjectConfig)

    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    model: ModelConfig = field(default_factory=ModelConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)

    checkpoint: CheckpointConfig = field(
        default_factory=CheckpointConfig
    )

    logging: LoggingConfig = field(default_factory=LoggingConfig)

    device: DeviceConfig = field(default_factory=DeviceConfig)


# ==============================================================================
# Global Configuration Instance
# ==============================================================================

CONFIG = Config()


# ==============================================================================
# Automatically Create Required Directories
# ==============================================================================

CONFIG.checkpoint.generator_dir.mkdir(parents=True, exist_ok=True)

CONFIG.checkpoint.discriminator_dir.mkdir(parents=True, exist_ok=True)

CONFIG.checkpoint.best_dir.mkdir(parents=True, exist_ok=True)