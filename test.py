import torch

from configs.config import CONFIG
from colorization.models.generator import Generator
from colorization.models.discriminator import PatchGANDiscriminator

device = CONFIG.device.device

generator = Generator().to(device)
discriminator = PatchGANDiscriminator().to(device)

x = torch.randn(
    2,
    CONFIG.dataset.input_channels,
    CONFIG.dataset.image_size,
    CONFIG.dataset.image_size,
).to(device)

with torch.no_grad():
    fake = generator(x)

print("Generator Output:", fake.shape)

with torch.no_grad():
    pred = discriminator(x, fake)

print("Discriminator Output:", pred.shape)