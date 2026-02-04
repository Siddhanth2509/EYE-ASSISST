import torch
import torch.nn as nn


class DRBinaryHead(nn.Module):
    """
    Binary classification head for Diabetic Retinopathy screening.
    
    Outputs a single logit for binary classification:
    - 0: NORMAL (no DR)
    - 1: DR (any stage of diabetic retinopathy)
    
    Uses a simple linear layer with optional dropout for regularization.
    """
    
    def __init__(
        self,
        feature_dim: int,
        dropout: float = 0.5
    ):
        """
        Initialize DR binary classification head.
        
        Args:
            feature_dim: Input feature dimension from backbone (e.g., 512 for ResNet18)
            dropout: Dropout probability for regularization (default: 0.5)
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor from backbone of shape (B, feature_dim)
        
        Returns:
            Logits of shape (B, 1) or (B,) after squeeze
        """
        x = self.dropout(x)
        logits = self.fc(x)
        return logits.squeeze()
