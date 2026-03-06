import torch
import torch.nn as nn


class DRSeverityHead(nn.Module):
    """
    DR severity head (grades 0–4).
    Takes backbone features and outputs 5-class logits.
    """

    def __init__(self, feature_dim: int, num_classes: int = 5):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim)
        Returns:
            logits: (B, 5)
        """
        return self.classifier(features)
