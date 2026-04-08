import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    """
    ResNet18 backbone for feature extraction.
    Outputs feature embeddings without any task-specific logic.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet18 with modern weights API
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

        # Feature dimension before classification head
        self.feature_dim = model.fc.in_features

        # Remove classification head
        model.fc = nn.Identity()

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, feature_dim)
        """
        return self.model(x)
