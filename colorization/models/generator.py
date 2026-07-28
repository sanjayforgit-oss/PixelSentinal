"""
Pix2Pix U-Net Generator for PixelSentinel.

Implements the generator described in:

    Isola et al.
    "Image-to-Image Translation with Conditional Adversarial Networks"
    CVPR 2017.

The generator follows an encoder-decoder U-Net architecture with
skip connections between corresponding encoder and decoder stages,
enhanced with an edge-sharpening module for high-frequency satellite detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.config import CONFIG
from colorization.models.blocks import (
    ConvBlock,
    DownBlock,
    UpBlock,
    FinalBlock,
)

__all__ = ["Generator"]


class EdgeSharpeningModule(nn.Module):
    """
    Explicit high-frequency detail and edge enhancement sub-module
    to sharpen satellite boundaries (roads, water bodies, agricultural borders).
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Depthwise-separable or fine convolutional residual path for edges
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual high-frequency injection
        return x + self.edge_conv(x)


class Generator(nn.Module):
    """
    Pix2Pix U-Net Generator with Edge Sharpening.

    Input:
        (B, input_channels, H, W)

    Output:
        (B, output_channels, H, W)
    """

    def __init__(self) -> None:
        super().__init__()

        in_channels = CONFIG.dataset.input_channels
        out_channels = CONFIG.dataset.output_channels
        features = CONFIG.model.generator_features

        # ==========================================================
        # Encoder
        # ==========================================================

        self.down1 = DownBlock(
            in_channels=in_channels,
            out_channels=features,
            normalization="none",
        )

        self.down2 = DownBlock(
            in_channels=features,
            out_channels=features * 2,
        )

        self.down3 = DownBlock(
            in_channels=features * 2,
            out_channels=features * 4,
        )

        self.down4 = DownBlock(
            in_channels=features * 4,
            out_channels=features * 8,
        )

        self.down5 = DownBlock(
            in_channels=features * 8,
            out_channels=features * 8,
        )

        self.down6 = DownBlock(
            in_channels=features * 8,
            out_channels=features * 8,
        )

        self.down7 = DownBlock(
            in_channels=features * 8,
            out_channels=features * 8,
        )

        # ==========================================================
        # Bottleneck
        # ==========================================================

        self.bottleneck = ConvBlock(
            in_channels=features * 8,
            out_channels=features * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            normalization="none",
            activation="relu",
            bias=True,
        )

        # ==========================================================
        # Decoder
        # ==========================================================

        self.up1 = UpBlock(
            in_channels=features * 8,
            out_channels=features * 8,
        )

        self.up2 = UpBlock(
            in_channels=features * 16,
            out_channels=features * 8,
        )

        self.up3 = UpBlock(
            in_channels=features * 16,
            out_channels=features * 8,
        )

        self.up4 = UpBlock(
            in_channels=features * 16,
            out_channels=features * 8,
        )

        self.up5 = UpBlock(
            in_channels=features * 16,
            out_channels=features * 4,
        )

        self.up6 = UpBlock(
            in_channels=features * 8,
            out_channels=features * 2,
        )

        self.up7 = UpBlock(
            in_channels=features * 4,
            out_channels=features,
        )

        # ==========================================================
        # Edge Sharpening & Super-Resolution Enhancement
        # ==========================================================
        self.edge_enhancer = EdgeSharpeningModule(channels=features * 2)

        self.final = FinalBlock(
            in_channels=features * 2,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input infrared image.

        Returns
        -------
        torch.Tensor
            Generated RGB image with sharp spatial boundaries.
        """

        # ==========================================================
        # Encoder
        # ==========================================================

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # ==========================================================
        # Bottleneck
        # ==========================================================

        bottleneck = self.bottleneck(d7)

        # ==========================================================
        # Decoder
        # ==========================================================

        u1 = self.up1(bottleneck)
        u1 = torch.cat((u1, d7), dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat((u2, d6), dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat((u3, d5), dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat((u4, d4), dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat((u5, d3), dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat((u6, d2), dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat((u7, d1), dim=1)

        # Apply high-frequency edge enhancement before final projection
        enhanced_features = self.edge_enhancer(u7)

        output = self.final(enhanced_features)

        return output