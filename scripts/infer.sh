#!/bin/bash
# Inference script

echo "Starting inference pipeline..."

# Set input and output paths
INPUT_PATH=${1:-"datasets/raw"}
OUTPUT_PATH=${2:-"outputs"}

python -m enhancement.inference --input $INPUT_PATH --output $OUTPUT_PATH
python -m colorization.inference --input $OUTPUT_PATH --output $OUTPUT_PATH
python -m detection.inference --input $OUTPUT_PATH --output $OUTPUT_PATH

echo "Inference completed!"