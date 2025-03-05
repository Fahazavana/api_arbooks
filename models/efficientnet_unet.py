import torch
import torch.nn as nn
from torchvision import models


class EfficientNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetUNet, self).__init__()
        # Charger un modèle EfficientNet pré-entraîné
        self.encoder = models.efficientnet_b3(pretrained=True)

        # Modifier la dernière couche pour correspondre au nombre de classes
        in_features = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.encoder(x)
