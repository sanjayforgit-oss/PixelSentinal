# PixelSentinel

Infrared image colorization and enhancement for improved object interpretation.

## Overview

PixelSentinel is a comprehensive system for processing, enhancing, and colorizing infrared satellite imagery from Landsat 8, Landsat 9, and Sentinel-2 sensors.

## Features

- **Image Enhancement**: SwinIR and RealESRGAN models for super-resolution
- **Colorization**: Pix2Pix and CycleGAN for realistic color mapping
- **Segmentation**: SegFormer, DeepLabV3, and UNet architectures
- **Object Detection**: YOLOv11 for feature detection
- **Validation**: Semantic, spectral, and hallucination checks
- **Evaluation**: PSNR, SSIM, FID, LPIPS metrics

## Quick Start

See [setup.md](docs/setup.md) for detailed installation instructions.

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Setup Guide](docs/setup.md)
- [References](docs/references.md)


#####################################################
# Create virtual environment
python -m venv .venv

# Activate

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
