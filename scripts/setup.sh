#!/bin/bash
# Setup script

echo "Setting up PixelSentinel..."

# Create directories
mkdir -p datasets/raw/landsat8
mkdir -p datasets/raw/landsat9
mkdir -p datasets/raw/sentinel2
mkdir -p datasets/processed/train
mkdir -p datasets/processed/val
mkdir -p datasets/processed/test
mkdir -p models/checkpoints
mkdir -p models/pretrained
mkdir -p models/exported
mkdir -p logs
mkdir -p outputs

echo "Directories created!"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup completed!"