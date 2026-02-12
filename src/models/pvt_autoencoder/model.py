import torch.nn as nn
from timm import create_model


class PVTv2B1Autoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        freeze_encoder: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        self.encoder = create_model(
            "pvt_v2_b1", pretrained=pretrained, features_only=True
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Bottleneck: 512x2x2 = 2048 -> latent_dim
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

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

    def forward(self, x):
        features = self.encoder(x)
        encoded = features[-1]  # (B, 512, H, W)
        z = self.to_latent(encoded)  # (B, latent_dim)
        spatial = self.from_latent(z)  # (B, 512, 2, 2)
        reconstructed = self.decoder(spatial)
        return reconstructed

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
