import torch
import torch.nn as nn


class SeverityLoss(nn.Module):
    """
    Class-Weighted CrossEntropyLoss for DR severity grading (0–4).

    Wraps nn.CrossEntropyLoss with optional per-class weights to handle
    severe class imbalance (e.g., 73.5% No DR vs 6.4% Mild).

    Interface:
        - Input:   logits (B, 5), targets (B,) LongTensor with values 0–4
        - Output:  scalar loss
        - Drop-in: same signature as nn.CrossEntropyLoss

    Usage:
        # Plain CE (no class weights)
        loss_fn = SeverityLoss()

        # Class-weighted CE (pass weights from datamodule.get_class_weights())
        loss_fn = SeverityLoss(class_weights=weights_tensor)
    """

    def __init__(self, class_weights: torch.Tensor = None, ignore_index: int = -100):
        """
        Args:
            class_weights: Optional (num_classes,) tensor of per-class weights.
                           Typically inverse-frequency weights from the training set.
                           If None, all classes are weighted equally (plain CE).
            ignore_index: Target value to exclude from loss computation.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, 5) raw unnormalized scores from severity head
            targets: (B,)   LongTensor with integer grades 0–4
        Returns:
            Scalar loss tensor (differentiable)
        """
        return self.ce(logits, targets)