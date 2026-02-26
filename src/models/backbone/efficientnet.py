import torch
import torch.nn as nn
from torchvision import models


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone for feature extraction.
    Outputs feature embeddings without any task-specific logic.

    Supported variants:
        - efficientnet_b0: 1280-dim features, ideal input 224px
        - efficientnet_b3: 1536-dim features, ideal input 300-384px

    Interface matches ResNetBackbone:
        - self.feature_dim: int
        - self.model: the underlying torchvision model
        - forward(x) -> (B, feature_dim)
    """

    VARIANTS = {
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
    }

    def __init__(self, variant: str = "efficientnet_b3", pretrained: bool = True):
        super().__init__()

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}'. "
                f"Choose from: {list(self.VARIANTS.keys())}"
            )

        factory_fn, weights_enum = self.VARIANTS[variant]
        weights = weights_enum if pretrained else None
        model = factory_fn(weights=weights)

        # EfficientNet classifier: Sequential(Dropout, Linear(feature_dim, 1000))
        # Extract feature_dim from the final classifier's Linear layer
        self.feature_dim = model.classifier[1].in_features

        # Remove classification head — replace with Identity
        model.classifier = nn.Identity()

        self.model = model
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, feature_dim)
        """
        return self.model(x)

    def get_finetune_layers(self):
        """
        Return the last few feature blocks for fine-tuning.

        EfficientNet has model.features (Sequential of MBConv blocks 0-8).
        Returns the last 2 blocks (blocks 7 and 8) — analogous to ResNet layer4.

        Returns:
            list of nn.Module: Feature blocks to unfreeze for fine-tuning.
        """
        features = self.model.features
        # Last 2 blocks of the feature extractor
        return [features[-1], features[-2]]
