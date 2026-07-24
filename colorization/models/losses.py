"""
Loss functions for PixelSentinel Pix2Pix training.

This module implements the adversarial and reconstruction losses used
during Pix2Pix training.

References
----------
Isola et al.
"Image-to-Image Translation with Conditional Adversarial Networks"
CVPR 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from configs.config import CONFIG

__all__ = [
    "Pix2PixLossOutput",
    "Pix2PixLoss",
]


@dataclass(slots=True)
class Pix2PixLossOutput:
    """
    Container for generator and discriminator losses.
    """

    generator_loss: Tensor
    discriminator_loss: Tensor
    generator_gan_loss: Tensor
    generator_l1_loss: Tensor
    discriminator_real_loss: Tensor
    discriminator_fake_loss: Tensor

    def as_dict(self) -> Dict[str, float]:
        """
        Convert losses to a Python dictionary.

        Returns
        -------
        Dict[str, float]
            Dictionary containing scalar loss values.
        """
        return {
            "generator_loss": float(self.generator_loss.detach().item()),
            "discriminator_loss": float(
                self.discriminator_loss.detach().item()
            ),
            "generator_gan_loss": float(
                self.generator_gan_loss.detach().item()
            ),
            "generator_l1_loss": float(
                self.generator_l1_loss.detach().item()
            ),
            "discriminator_real_loss": float(
                self.discriminator_real_loss.detach().item()
            ),
            "discriminator_fake_loss": float(
                self.discriminator_fake_loss.detach().item()
            ),
        }


class Pix2PixLoss(nn.Module):
    """
    Combined Pix2Pix loss.

    Generator Loss

        L_G = GAN Loss + λ * L1 Loss

    Discriminator Loss

        L_D = 0.5 * (Real Loss + Fake Loss)
    """

    def __init__(self) -> None:
        """
        Initialize loss functions.
        """
        super().__init__()

        self.lambda_l1: float = CONFIG.training.lambda_l1

        self.gan_loss = nn.BCEWithLogitsLoss()

        self.l1_loss = nn.L1Loss()

    @staticmethod
    def _real_labels(prediction: Tensor) -> Tensor:
        """
        Create a tensor of real labels.

        Parameters
        ----------
        prediction:
            PatchGAN prediction tensor.

        Returns
        -------
        Tensor
            Tensor filled with ones.
        """
        return torch.ones_like(
            prediction,
            device=prediction.device,
            dtype=prediction.dtype,
        )

    @staticmethod
    def _fake_labels(prediction: Tensor) -> Tensor:
        """
        Create a tensor of fake labels.

        Parameters
        ----------
        prediction:
            PatchGAN prediction tensor.

        Returns
        -------
        Tensor
            Tensor filled with zeros.
        """
        return torch.zeros_like(
            prediction,
            device=prediction.device,
            dtype=prediction.dtype,
        )
    
    def generator_loss(
        self,
        fake_prediction: Tensor,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the generator loss.

        The generator minimizes:

            L_G = L_GAN + λ * L1

        Args:
            fake_prediction:
                PatchGAN prediction for generated images.

            generated_image:
                Generated RGB image.

            target_image:
                Ground-truth RGB image.

        Returns:
            Tuple containing:

            - Total generator loss
            - GAN loss
            - L1 reconstruction loss
        """
        real_labels = self._real_labels(fake_prediction)

        gan_loss = self.gan_loss(
            fake_prediction,
            real_labels,
        )

        l1_loss = self.l1_loss(
            generated_image,
            target_image,
        )

        total_loss = gan_loss + self.lambda_l1 * l1_loss

        return total_loss, gan_loss, l1_loss

    def discriminator_loss(
        self,
        real_prediction: Tensor,
        fake_prediction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the discriminator loss.

        Args:
            real_prediction:
                PatchGAN prediction for real image pairs.

            fake_prediction:
                PatchGAN prediction for generated image pairs.

        Returns:
            Tuple containing:

            - Total discriminator loss
            - Real loss
            - Fake loss
        """
        real_labels = self._real_labels(real_prediction)
        fake_labels = self._fake_labels(fake_prediction)

        real_loss = self.gan_loss(
            real_prediction,
            real_labels,
        )

        fake_loss = self.gan_loss(
            fake_prediction,
            fake_labels,
        )

        total_loss = 0.5 * (real_loss + fake_loss)

        return total_loss, real_loss, fake_loss

    def forward(
        self,
        real_prediction: Tensor,
        fake_prediction_for_discriminator: Tensor,
        fake_prediction_for_generator: Tensor,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> Pix2PixLossOutput:
        """
        Compute all Pix2Pix losses.

        Args:
            real_prediction:
                Discriminator prediction for real image pairs.

            fake_prediction_for_discriminator:
                Discriminator prediction for fake image pairs
                (computed using detached generator output).

            fake_prediction_for_generator:
                Discriminator prediction for fake image pairs
                (computed without detaching the generator output).

            generated_image:
                RGB image produced by the generator.

            target_image:
                Ground-truth RGB image.

        Returns:
            Pix2PixLossOutput containing all losses required for
            optimization and logging.
        """
        generator_loss, gan_loss, l1_loss = self.generator_loss(
            fake_prediction=fake_prediction_for_generator,
            generated_image=generated_image,
            target_image=target_image,
        )

        discriminator_loss, real_loss, fake_loss = (
            self.discriminator_loss(
                real_prediction=real_prediction,
                fake_prediction=fake_prediction_for_discriminator,
            )
        )

        return Pix2PixLossOutput(
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            generator_gan_loss=gan_loss,
            generator_l1_loss=l1_loss,
            discriminator_real_loss=real_loss,
            discriminator_fake_loss=fake_loss,
        )