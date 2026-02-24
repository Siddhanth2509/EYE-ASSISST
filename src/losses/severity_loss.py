import torch
import torch.nn as nn
import torch.nn.functional as F


class SeverityLoss(nn.Module):
    """
    Ordinal-Aware Distance-Weighted CrossEntropyLoss for DR severity grading (0–4).

    Unlike plain CrossEntropyLoss which treats all misclassifications equally,
    this loss penalizes predictions proportionally to how far they are from
    the true grade:

        weight_i = 1 + alpha * |argmax(logits_i) - target_i|
        loss = mean( CE_i * weight_i )

    This means predicting Grade 0 when truth is Grade 4 (distance=4) gets
    penalized much more than predicting Grade 3 for Grade 4 (distance=1).

    Interface:
        - Input:   logits (B, 5), targets (B,) LongTensor with values 0–4
        - Output:  scalar loss
        - Drop-in: same signature as nn.CrossEntropyLoss

    Notes:
        - Does NOT apply softmax manually (F.cross_entropy handles that)
        - argmax is non-differentiable but that's fine — it only scales the
          gradient magnitude, not direction. CE gradient still flows normally
          through logits.
        - Masking via ignore_index is handled before any computation.
    """

    def __init__(self, alpha: float = 0.5, ignore_index: int = -100, num_classes: int = 5):
        """
        Args:
            alpha: Ordinal distance multiplier. Higher = stronger penalty for
                   far-off predictions. Recommended range: 0.3–1.0.
                   At alpha=0 this reduces to plain CrossEntropyLoss.
            ignore_index: Target value to exclude from loss computation.
            num_classes: Number of severity grades (default 5 for DR 0–4).
        """
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 5) raw unnormalized scores from severity head
            targets: (B,)   LongTensor with integer grades 0–4
        Returns:
            Scalar loss tensor (differentiable)
        """
        # --- Input validation ---
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape (B, C), got {logits.shape}")
        if targets.ndim != 1:
            raise ValueError(f"Expected targets shape (B,), got {targets.shape}")

        # --- Mask invalid targets ---
        valid_mask = targets != self.ignore_index
        num_valid = valid_mask.sum().item()

        if num_valid == 0:
            # No valid samples — return zero loss that still has grad_fn
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits[valid_mask]       # (N, 5)
        targets_valid = targets[valid_mask]     # (N,)

        # --- Per-sample cross entropy (no reduction) ---
        ce_loss = F.cross_entropy(logits_valid, targets_valid, reduction="none")  # (N,)

        # --- Ordinal distance weighting ---
        preds = torch.argmax(logits_valid, dim=1)                   # (N,) non-differentiable
        distance = torch.abs(preds.float() - targets_valid.float()) # (N,) float

        # Clamp distance to [0, num_classes-1] for safety
        distance = distance.clamp(max=self.num_classes - 1)

        # Weight: 1.0 for correct predictions, up to (1 + alpha*(K-1)) for worst case
        weights = 1.0 + self.alpha * distance  # (N,)

        # --- Weighted mean reduction ---
        weighted_loss = (ce_loss * weights).sum() / num_valid

        return weighted_loss