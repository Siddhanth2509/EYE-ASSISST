import torch
import torch.nn as nn


class DRBinaryHead(nn.Module):
    """
    Binary DR screening head.
    Takes backbone features and outputs a single logit.
    """

    def __init__(self, feature_dim: int):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, feature_dim)

        Returns:
            logits: Tensor of shape (B, 1)
        """
        return self.classifier(features)
