from torchsummary import summary
from src.models.cnn_autoencoder.model import CNNAutoEncoder
from src.models.resnet18_autoencoder.model import ResNet18Autoencoder
from src.models.pvt_autoencoder.model import PVTv2B1Autoencoder

model = CNNAutoEncoder(latent_dim=128)
summary(model, (3, 64, 64))

model = ResNet18Autoencoder(latent_dim=128, pretrained=False)
summary(model, (3, 64, 64))

model = PVTv2B1Autoencoder(latent_dim=128, pretrained=False)
summary(model, (3, 64, 64))
