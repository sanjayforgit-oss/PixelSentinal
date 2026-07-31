"""
Reusable neural network building blocks for PixelSentinel.

This module provides modular convolutional blocks that are shared by both the
Generator and PatchGAN Discriminator architectures.
"""

from __future__ import annotations

from typing import Callable, Optional, Type

import torch
import torch.nn as nn

from configs.config import CONFIG

__all__ = [
    "Identity",
    "get_normalization",
    "get_activation",
    "ConvBlock",
    "DownBlock",
    "UpBlock",
    "FinalBlock",
]


class Identity(nn.Module):
    """
    Identity layer.

    Returns the input tensor unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Same tensor.
        """
        return x


def get_normalization(
    normalization: Optional[str],
    num_features: int,
) -> nn.Module:
    """
    Create a normalization layer.

    Supported values:
        - "batch"
        - "instance"
        - "none"

    Args:
        normalization:
            Name of normalization layer.
        num_features:
            Number of channels.

    Returns:
        Instantiated normalization module.

    Raises:
        ValueError:
            If normalization type is unsupported.
    """
    normalization = (
        CONFIG.model.normalization
        if normalization is None
        else normalization
    )

    normalization = normalization.lower()

    if normalization == "batch":
        return nn.BatchNorm2d(num_features)

    if normalization == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)

    if normalization == "none":
        return Identity()

    raise ValueError(f"Unsupported normalization: {normalization}")


def get_activation(name: str) -> nn.Module:
    """
    Create an activation function.

    Supported activations:
        - relu
        - leakyrelu
        - tanh
        - sigmoid

    Args:
        name:
            Activation name.

    Returns:
        Activation module.

    Raises:
        ValueError:
            If activation is unsupported.
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)

    if name == "tanh":
        return nn.Tanh()

    if name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Unsupported activation: {name}")

class ConvBlock(nn.Module):
    """
    Standard convolution block.

    Structure:

        Conv2d
        -> Normalization
        -> Activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        normalization: Optional[str] = None,
        activation: str = "relu",
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # Automatically determine bias
        if bias is None:
            norm_type = (
                CONFIG.model.normalization
                if normalization is None
                else normalization
            )

            bias = norm_type.lower() == "none"

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            get_normalization(normalization, out_channels),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block for encoder.

    Uses ConvBlock with LeakyReLU activation by default.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        """
        Initialize DownBlock.

        Args:
            in_channels:
                Input channels.
            out_channels:
                Output channels.
            normalization:
                Normalization type.
            kernel_size:
                Kernel size.
            stride:
                Stride.
            padding:
                Padding.
            bias:
                Conv bias.
        """
        super().__init__()

        self.block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            normalization=normalization,
            activation="leakyrelu",
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block using resize-convolution to avoid checkerboard artifacts.

    Structure:

        Upsample (Bilinear)
        -> Conv2d
        -> Normalization
        -> ReLU
        -> Optional Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if bias is None:
            norm_type = (
                CONFIG.model.normalization
                if normalization is None
                else normalization
            )
            bias = norm_type.lower() == "none"

        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            get_normalization(normalization, out_channels),
            get_activation("relu"),
        ]

        if bool(CONFIG.model.use_dropout):
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class FinalBlock(nn.Module):
    """
    Final generator output layer.

    Structure:

        ConvTranspose2d
        -> Tanh
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Initialize FinalBlock.

        Args:
            in_channels:
                Input channels.
            out_channels:
                Output channels (RGB = 3).
            kernel_size:
                Kernel size.
            stride:
                Stride.
            padding:
                Padding.
            bias:
                ConvTranspose bias.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            get_activation("tanh"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Input tensor.

        Returns:
            RGB image tensor in [-1, 1].
        """
        return self.block(x)