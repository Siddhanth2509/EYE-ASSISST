import torch
import torch.nn as nn

from .backbone.resnet import ResNetBackbone
from .heads.dr_binary import DRBinaryHead
from .heads.dr_severity import DRSeverityHead

class MultiTaskModel(nn.Module):
    """
    Multi-task model with a shared backbone and task-specific heads.
    """

    def __init__(self, backbone: str = "resnet50", backbone_pretrained: bool = True):
        super().__init__()

        # Shared backbone (currently only resnet50 supported)
        self.backbone = ResNetBackbone(pretrained=backbone_pretrained)

        # Heads (start with DR binary only)
        self.dr_binary_head = DRBinaryHead(
            feature_dim=self.backbone.feature_dim
        )

        # DR severity head (grades 0-4)
        self.dr_severity_head = DRSeverityHead(
            feature_dim=self.backbone.feature_dim
        )

        # Placeholders for future heads
        self.multilabel_head = None

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            outputs: Dict of logits per head
        """
        features = self.backbone(x)

        outputs = {
            "dr_binary": self.dr_binary_head(features)
        }

        # Future extensions (kept explicit)
        if self.dr_severity_head is not None:
            outputs["dr_severity"] = self.dr_severity_head(features)

        if self.multilabel_head is not None:
            outputs["multilabel"] = self.multilabel_head(features)

        return outputs
