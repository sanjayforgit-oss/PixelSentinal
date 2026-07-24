"""
Pix2Pix U-Net Generator for PixelSentinel.

This module implements the generator network used for infrared-to-RGB
image translation. The architecture follows the original Pix2Pix U-Net
design with encoder-decoder skip connections.

References:
    - Isola et al., "Image-to-Image Translation with Conditional
      Adversarial Networks", CVPR 2017.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from configs.config import CONFIG
from colorization.models.blocks import (
    ConvBlock,
    DownBlock,
    FinalBlock,
    UpBlock,
)

__all__ = ["Generator"]


class Generator(nn.Module):
    """
    Pix2Pix U-Net Generator.

    The network consists of

    Encoder
        8 downsampling stages

    Bottleneck

    Decoder
        8 upsampling stages with skip connections.

    Input:
        (B, C_in, H, W)

    Output:
        (B, C_out, H, W)

    where

        C_in  = CONFIG.dataset.input_channels
        C_out = CONFIG.dataset.output_channels
    """

    def __init__(self) -> None:
        """
        Initialize Generator.
        """
        super().__init__()

        in_channels = CONFIG.dataset.input_channels
        out_channels = CONFIG.dataset.output_channels
        features = CONFIG.model.generator_features

        #
        # -----------------------------
        # Encoder
        # -----------------------------
        #

        # First layer does not use normalization
        self.down1 = ConvBlock(
            in_channels=in_channels,
            out_channels=features,
            normalization="none",
            activation="leakyrelu",
            bias=True,
        )

        self.down2 = DownBlock(
            features,
            features * 2,
        )

        self.down3 = DownBlock(
            features * 2,
            features * 4,
        )

        self.down4 = DownBlock(
            features * 4,
            features * 8,
        )

        self.down5 = DownBlock(
            features * 8,
            features * 8,
        )

        self.down6 = DownBlock(
            features * 8,
            features * 8,
        )

        self.down7 = DownBlock(
            features * 8,
            features * 8,
        )

        self.down8 = DownBlock(
            features * 8,
            features * 8,
        )

        #
        # -----------------------------
        # Bottleneck
        # -----------------------------
        #

        self.bottleneck = ConvBlock(
            in_channels=features * 8,
            out_channels=features * 8,
            normalization="none",
            activation="relu",
            bias=True,
        )

        #
        # -----------------------------
        # Decoder
        # -----------------------------
        #

        self.up1 = UpBlock(
            features * 8,
            features * 8,
        )

        self.up2 = UpBlock(
            features * 16,
            features * 8,
        )

        self.up3 = UpBlock(
            features * 16,
            features * 8,
        )

        self.up4 = UpBlock(
            features * 16,
            features * 8,
        )

        self.up5 = UpBlock(
            features * 16,
            features * 4,
        )

        self.up6 = UpBlock(
            features * 8,
            features * 2,
        )

        self.up7 = UpBlock(
            features * 4,
            features,
        )

        self.final = FinalBlock(
            in_channels=features * 2,
            out_channels=out_channels,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Pix2Pix U-Net Generator.

        Args:
            x:
                Input tensor of shape
                (B, CONFIG.dataset.input_channels, H, W).

        Returns:
            Generated RGB tensor of shape
            (B, CONFIG.dataset.output_channels, H, W).
        """

        #
        # -----------------------------
        # Encoder
        # -----------------------------
        #

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        #
        # -----------------------------
        # Bottleneck
        # -----------------------------
        #

        bottleneck = self.bottleneck(d8)

        #
        # -----------------------------
        # Decoder with Skip Connections
        # -----------------------------
        #

        u1 = self.up1(bottleneck)
        u1 = torch.cat((u1, d8), dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat((u2, d7), dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat((u3, d6), dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat((u4, d5), dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat((u5, d4), dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat((u6, d3), dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat((u7, d2), dim=1)

        output = self.final(u7)

        return output