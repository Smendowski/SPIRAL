import torch.nn as nn
from torchvision import models


class ResNet18Autoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        freeze_encoder: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        resnet = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # -> (512, 2, 2)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Bottleneck: 512x2x2 = 2048 -> latent_dim
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (512, 2, 2) -> (512, 1, 1)
            nn.Flatten(),  # → (512,)
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        # back from latent to spatial
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 512 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (512, 2, 2)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        z = self.to_latent(features)
        return z

    def decode(self, z):
        spatial = self.from_latent(z)
        return self.decoder(spatial)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
