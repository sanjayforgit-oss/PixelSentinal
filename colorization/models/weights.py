"""
Weight initialization utilities for PixelSentinel.

This module provides reusable weight initialization functions for the
Pix2Pix Generator and PatchGAN Discriminator.

Reference:
    Isola et al., "Image-to-Image Translation with Conditional
    Adversarial Networks", CVPR 2017.

The original Pix2Pix implementation initializes convolutional weights
from a Normal distribution with mean=0.0 and std=0.02.
"""

from __future__ import annotations

import logging

import torch.nn as nn

__all__ = [
    "initialize_weights",
]


LOGGER = logging.getLogger(__name__)


def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights following the Pix2Pix paper.

    Initialization strategy:

    - Conv2d:
        Normal(mean=0.0, std=0.02)

    - ConvTranspose2d:
        Normal(mean=0.0, std=0.02)

    - BatchNorm2d:
        Weight -> Normal(mean=1.0, std=0.02)
        Bias   -> Constant(0.0)

    - InstanceNorm2d:
        Initialized only if affine=True.

    Args:
        model:
            PyTorch model to initialize.
    """

    for module in model.modules():

        #
        # Convolution Layers
        #
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        #
        # Batch Normalization
        #
        elif isinstance(module, nn.BatchNorm2d):

            if module.weight is not None:
                nn.init.normal_(module.weight, mean=1.0, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        #
        # Instance Normalization
        #
        elif isinstance(module, nn.InstanceNorm2d):

            if module.affine:

                if module.weight is not None:
                    nn.init.normal_(module.weight, mean=1.0, std=0.02)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    LOGGER.info(
        "Weights initialized using Pix2Pix initialization strategy."
    )