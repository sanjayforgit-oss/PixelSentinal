"""
Loss functions for PixelSentinel Pix2Pix training.

Implements adversarial, reconstruction (L1), VGG-19 perceptual realism loss, 
and NDVI/NDWI spectral index priors for geographical consistency.

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
import torchvision.models as models
from torch import Tensor

from configs.config import CONFIG

__all__ = [
    "Pix2PixLossOutput",
    "Pix2PixLoss",
]


class VGGPerceptualLoss(nn.Module):
    """
    Computes VGG-19 feature-space perceptual loss to encourage sharp 
    textural and structural realism in satellite colorization outputs.
    """
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        # Load pre-trained VGG-19 features up to layer 16 (relu3_4)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.to(device)

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        # Normalize from [-1, 1] to [0, 1] if needed, or use directly depending on transform
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return nn.functional.l1_loss(gen_features, target_features)


@dataclass(slots=True)
class Pix2PixLossOutput:
    """
    Container for generator and discriminator losses, including perceptual and spectral penalties.
    """

    generator_loss: Tensor
    discriminator_loss: Tensor
    generator_gan_loss: Tensor
    generator_l1_loss: Tensor
    generator_perceptual_loss: Tensor
    generator_spectral_loss: Tensor
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
            "discriminator_loss": float(self.discriminator_loss.detach().item()),
            "generator_gan_loss": float(self.generator_gan_loss.detach().item()),
            "generator_l1_loss": float(self.generator_l1_loss.detach().item()),
            "generator_perceptual_loss": float(self.generator_perceptual_loss.detach().item()),
            "generator_spectral_loss": float(self.generator_spectral_loss.detach().item()),
            "discriminator_real_loss": float(self.discriminator_real_loss.detach().item()),
            "discriminator_fake_loss": float(self.discriminator_fake_loss.detach().item()),
        }


class Pix2PixLoss(nn.Module):
    """
    Combined multi-objective Pix2Pix loss with Perceptual & Spectral Priors.

    Generator Loss

        L_G = GAN Loss + λ_l1 * L1 Loss + λ_perc * Perceptual Loss + λ_spec * Spectral Loss

    Discriminator Loss

        L_D = 0.5 * (Real Loss + Fake Loss)
    """

    def __init__(self) -> None:
        """
        Initialize loss functions and weights.
        """
        super().__init__()

        self.lambda_l1: float = getattr(CONFIG.training, "lambda_l1", 100.0)
        self.lambda_perceptual: float = getattr(CONFIG.training, "lambda_perceptual", 10.0)
        self.lambda_spectral: float = getattr(CONFIG.training, "lambda_spectral", 5.0)

        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Will be moved to device on .to(device) call
        self.perceptual_loss_fn = None  
        self._vgg_initialized = False

    def _init_vgg(self, device: torch.device) -> None:
        if not self._vgg_initialized:
            try:
                self.perceptual_loss_fn = VGGPerceptualLoss(device)
                self._vgg_initialized = True
            except Exception:
                # Fallback dummy if offline or weights unavailable
                pass

    @staticmethod
    def _real_labels(prediction: Tensor) -> Tensor:
        return torch.ones_like(prediction, device=prediction.device, dtype=prediction.dtype)

    @staticmethod
    def _fake_labels(prediction: Tensor) -> Tensor:
        return torch.zeros_like(prediction, device=prediction.device, dtype=prediction.dtype)

    def _compute_spectral_priors(
        self,
        input_image: Tensor,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> Tensor:
        """
        Computes NDVI (Vegetation) and NDWI (Water) consistency penalties
        using input NIR channel and RGB spectral bands.
        
        Input bands: [NIR (B5), SWIR1 (B6), SWIR2 (B7), Thermal (B10)]
        Target/Generated bands: [Red (B4), Green (B3), Blue (B2)] (or RGB)
        """
        # Extract NIR from input (Channel 0)
        nir = input_image[:, 0:1, :, :]
        
        # Extract Red from target/generated (Channel 0 of RGB target)
        red_gen = generated_image[:, 0:1, :, :]
        red_tgt = target_image[:, 0:1, :, :]
        
        # Extract Green from target/generated (Channel 1 of RGB target)
        green_gen = generated_image[:, 1:2, :, :]
        green_tgt = target_image[:, 1:2, :, :]

        # Approximate NDVI consistency: (NIR - Red) / (NIR + Red + 1e-6)
        ndvi_gen = (nir - red_gen) / (torch.abs(nir + red_gen) + 1e-6)
        ndvi_tgt = (nir - red_tgt) / (torch.abs(nir + red_tgt) + 1e-6)
        ndvi_loss = nn.functional.l1_loss(ndvi_gen, ndvi_tgt)

        # Approximate NDWI consistency using Green vs NIR: (Green - NIR) / (Green + NIR + 1e-6)
        ndwi_gen = (green_gen - nir) / (torch.abs(green_gen + nir) + 1e-6)
        ndwi_tgt = (green_tgt - nir) / (torch.abs(green_tgt + nir) + 1e-6)
        ndwi_loss = nn.functional.l1_loss(ndwi_gen, ndwi_tgt)

        return ndvi_loss + ndwi_loss

    def generator_loss(
        self,
        input_image: Tensor,
        fake_prediction: Tensor,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the generator loss including GAN, L1, Perceptual, and Spectral priors.
        """
        device = fake_prediction.device
        if not self._vgg_initialized:
            self._init_vgg(device)

        real_labels = self._real_labels(fake_prediction)

        gan_loss = self.gan_loss(fake_prediction, real_labels)
        l1_loss = self.l1_loss(generated_image, target_image)

        # Perceptual Loss
        if self.perceptual_loss_fn is not None:
            perceptual_loss = self.perceptual_loss_fn(generated_image, target_image)
        else:
            perceptual_loss = torch.tensor(0.0, device=device)

        # Spectral Prior Loss (NDVI & NDWI)
        spectral_loss = self._compute_spectral_priors(input_image, generated_image, target_image)

        total_loss = (
            gan_loss 
            + (self.lambda_l1 * l1_loss) 
            + (self.lambda_perceptual * perceptual_loss) 
            + (self.lambda_spectral * spectral_loss)
        )

        return total_loss, gan_loss, l1_loss, perceptual_loss, spectral_loss

    def discriminator_loss(
        self,
        real_prediction: Tensor,
        fake_prediction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        real_labels = self._real_labels(real_prediction)
        fake_labels = self._fake_labels(fake_prediction)

        real_loss = self.gan_loss(real_prediction, real_labels)
        fake_loss = self.gan_loss(fake_prediction, fake_labels)

        total_loss = 0.5 * (real_loss + fake_loss)
        return total_loss, real_loss, fake_loss

    def forward(
        self,
        input_image: Tensor,
        real_prediction: Tensor,
        fake_prediction_for_discriminator: Tensor,
        fake_prediction_for_generator: Tensor,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> Pix2PixLossOutput:
        """
        Compute all Pix2Pix losses including advanced hackathon constraints.
        """
        generator_loss, gan_loss, l1_loss, perceptual_loss, spectral_loss = self.generator_loss(
            input_image=input_image,
            fake_prediction=fake_prediction_for_generator,
            generated_image=generated_image,
            target_image=target_image,
        )

        discriminator_loss, real_loss, fake_loss = self.discriminator_loss(
            real_prediction=real_prediction,
            fake_prediction=fake_prediction_for_discriminator,
        )

        return Pix2PixLossOutput(
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            generator_gan_loss=gan_loss,
            generator_l1_loss=l1_loss,
            generator_perceptual_loss=perceptual_loss,
            generator_spectral_loss=spectral_loss,
            discriminator_real_loss=real_loss,
            discriminator_fake_loss=fake_loss,
        )