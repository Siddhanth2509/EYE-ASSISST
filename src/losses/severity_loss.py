import torch
import torch.nn as nn


class SeverityLoss(nn.Module):
    """
    Structured Severity Loss for DR grading (0–4)

    Features:
    - Uses CrossEntropyLoss internally
    - Supports optional masking for invalid labels
    - Stable and safe for Stage-2 training

    Expected:
        logits: Tensor (B, 5)
        targets: Tensor (B,)
    """

    def __init__(self, ignore_index: int = -1):
        super(SeverityLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: shape (batch_size, num_classes=5)
            targets: shape (batch_size,)
        """

        if logits.dim() != 2:
            raise ValueError("Severity logits must be 2D (B, C)")

        if targets.dim() != 1:
            raise ValueError("Severity targets must be 1D (B,)")

        loss = self.ce(logits, targets)

        return loss
