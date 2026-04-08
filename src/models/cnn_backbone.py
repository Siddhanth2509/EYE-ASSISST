# src/models/cnn_backbone.py

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CNNBackbone(nn.Module):
    """
    CNN backbone for binary DR classification using pretrained models.
    
    Supports multiple architectures optimized for medical image classification:
    - ResNet18/50: Robust, widely used, good for transfer learning
    - EfficientNet-B0: Efficient, good accuracy-to-parameters ratio
    - DenseNet121: Good feature extraction for medical images
    
    All architectures are pretrained on ImageNet and fine-tuned for binary classification.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        num_classes: int = 1,
    ):
        """
        Initialize CNN backbone.
        
        Args:
            backbone: Architecture name ('resnet18', 'resnet50', 'efficientnet_b0', 'densenet121')
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (1 for binary classification)
        """
        super(CNNBackbone, self).__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained
        
        # Stage-1 lock: Only resnet18 is allowed for Stage-1 screening
        # Other backbones are reserved for future experiments
        if backbone != "resnet18":
            raise ValueError(
                "Only resnet18 is allowed for Stage-1 screening. "
                "Other backbones are reserved for future experiments."
            )
        
        # Load backbone architecture using modern torchvision API
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = models.resnet18(weights=weights)
            # Replace final fully connected layer
            num_features = backbone_model.fc.in_features
            backbone_model.fc = nn.Linear(num_features, num_classes)
            self.features = nn.Sequential(*list(backbone_model.children())[:-1])
            self.classifier = backbone_model.fc
            
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = models.resnet50(weights=weights)
            num_features = backbone_model.fc.in_features
            backbone_model.fc = nn.Linear(num_features, num_classes)
            self.features = nn.Sequential(*list(backbone_model.children())[:-1])
            self.classifier = backbone_model.fc
            
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = models.efficientnet_b0(weights=weights)
            num_features = backbone_model.classifier[1].in_features
            self.features = backbone_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, num_classes)
            )
            
        elif backbone == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = models.densenet121(weights=weights)
            num_features = backbone_model.classifier.in_features
            backbone_model.classifier = nn.Linear(num_features, num_classes)
            self.features = backbone_model.features
            self.classifier = backbone_model.classifier
            
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: 'resnet18', 'resnet50', 'efficientnet_b0', 'densenet121'"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features
        if self.backbone_name in ['resnet18', 'resnet50']:
            x = self.features(x)
            x = torch.flatten(x, 1)  # Flatten for FC layer
        elif self.backbone_name == 'efficientnet_b0':
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif self.backbone_name == 'densenet121':
            features = self.features(x)
            x = nn.functional.relu(features, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        
        # Classification head
        x = self.classifier(x)
        
        return x
    
    def get_num_features(self) -> int:
        """
        Reserved for future multi-head / multi-disease extensions.
        Not used in Phase 2B.
        
        Returns:
            Number of input features to the classifier
        """
        if self.backbone_name in ['resnet18', 'resnet50']:
            return self.classifier.in_features
        elif self.backbone_name == 'efficientnet_b0':
            return self.classifier[1].in_features
        elif self.backbone_name == 'densenet121':
            return self.classifier.in_features
        return None


def create_backbone(
    backbone: str = 'resnet18',
    pretrained: bool = True,
) -> CNNBackbone:
    """
    Convenience function to create a CNN backbone.
    
    Args:
        backbone: Architecture name
        pretrained: Whether to use pretrained weights
    
    Returns:
        CNNBackbone instance
    """
    return CNNBackbone(
        backbone=backbone,
        pretrained=pretrained,
        num_classes=1,  # Binary classification
    )
