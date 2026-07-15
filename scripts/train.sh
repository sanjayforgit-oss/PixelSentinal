#!/bin/bash
# Training script

echo "Starting model training..."

# Train enhancement model
python -m enhancement.train --config configs/model_config.yaml

# Train colorization model
python -m colorization.train --config configs/model_config.yaml

# Train segmentation model
python -m segmentation.train --config configs/model_config.yaml

echo "Training completed!"