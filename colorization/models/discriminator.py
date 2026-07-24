"""
PatchGAN Discriminator for PixelSentinel.

This module implements the 70×70 PatchGAN discriminator proposed in the
Pix2Pix paper.

Reference:
    Isola et al.
    "Image-to-Image Translation with Conditional Adversarial Networks"
    CVPR 2017.

The discriminator receives the concatenation of the input infrared image
and the corresponding RGB image (real or generated) and predicts a matrix
of patch-wise real/fake probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import CONFIG
from colorization.models.blocks import ConvBlock, DownBlock

__all__ = ["PatchGANDiscriminator"]


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Input
    -----
    Infrared image:
        (B, Cin, H, W)

    RGB image:
        (B, Cout, H, W)

    Concatenated input:
        (B, Cin + Cout, H, W)

    Output
    ------
    Patch prediction map:
        (B, 1, H', W')
    """

    def __init__(self) -> None:
        """
        Initialize the PatchGAN discriminator.
        """
        super().__init__()

        in_channels = (
            CONFIG.dataset.input_channels
            + CONFIG.dataset.output_channels
        )

        features = CONFIG.model.discriminator_features

        #
        # Initial layer
        #
        # Pix2Pix does not apply normalization
        # on the first discriminator layer.
        #
        self.initial = ConvBlock(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=4,
            stride=2,
            padding=1,
            normalization="none",
            activation="leakyrelu",
            bias=True,
        )

        #
        # Feature extraction
        #
        self.down1 = DownBlock(
            in_channels=features,
            out_channels=features * 2,
        )

        self.down2 = DownBlock(
            in_channels=features * 2,
            out_channels=features * 4,
        )

        #
        # Last encoder stage
        #
        # Uses stride=1 exactly as in Pix2Pix.
        #
        self.down3 = DownBlock(
            in_channels=features * 4,
            out_channels=features * 8,
            stride=1,
        )

        #
        # Final prediction layer
        #
        self.classifier = nn.Conv2d(
            in_channels=features * 8,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(
        self,
        infrared: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the PatchGAN discriminator.

        The discriminator operates on the channel-wise concatenation of the
        infrared input image and its corresponding RGB image (real or
        generated), producing a patch-level authenticity map.

        Args:
            infrared:
                Infrared input tensor of shape
                (B, CONFIG.dataset.input_channels, H, W).

            rgb:
                RGB target or generated tensor of shape
                (B, CONFIG.dataset.output_channels, H, W).

        Returns:
            Patch authenticity logits of shape
            (B, 1, H_out, W_out).

        Raises:
            ValueError:
                If the spatial dimensions of the two inputs do not match.
        """

        if infrared.shape[0] != rgb.shape[0]:
            raise ValueError(
                "Infrared and RGB tensors must have the same batch size."
            )

        if infrared.shape[2:] != rgb.shape[2:]:
            raise ValueError(
                "Infrared and RGB tensors must have identical spatial dimensions."
            )

        x = torch.cat((infrared, rgb), dim=1)

        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        return self.classifier(x)