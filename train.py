#!/usr/bin/env python3
"""
Training script for binary DR classification.

Usage:
    python train.py --data_root Data/splits/fundus --epochs 50
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models

from src.data.datamodule import FundusDataModule
from src.training.train_binary import BinaryTrainer
from src.utils.seed import set_seed


def get_model(model_name: str = 'resnet18', pretrained: bool = True):
    """
    Get model architecture.
    
    Args:
        model_name: Model name ('resnet18', 'resnet50', 'efficientnet_b0', etc.)
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model with modified final layer for binary classification
    """
    import torchvision
    # Check torchvision version for compatibility
    tv_version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    
    # Use weights parameter for torchvision >= 0.13.0, pretrained for older versions
    if tv_version >= (0, 13):
        weights = 'IMAGENET1K_V1' if pretrained else None
        use_weights = True
    else:
        use_weights = False
    
    if model_name == 'resnet18':
        if use_weights:
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'resnet50':
        if use_weights:
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'efficientnet_b0':
        if use_weights:
            # EfficientNet uses different weights parameter
            weights = 'IMAGENET1K_V1' if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        else:
            model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train binary DR classifier')
    parser.add_argument(
        '--data_root',
        type=str,
        default='Data/splits/fundus',
        help='Path to data root (should contain eyepacs/ and aptos/ folders)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50', 'efficientnet_b0'],
        help='Model architecture'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Input image size'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no_class_weights',
        action='store_true',
        help='Disable class weighting for imbalanced data'
    )
    parser.add_argument(
        '--no_pretrained',
        action='store_true',
        help='Train from scratch (no pretrained weights)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models/binary_dr',
        help='Directory to save models'
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup data module
    print("Setting up data module...")
    data_module = FundusDataModule(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()
    
    # Setup model
    print(f"Loading model: {args.model}")
    model = get_model(
        model_name=args.model,
        pretrained=not args.no_pretrained
    )
    
    # Setup trainer
    trainer = BinaryTrainer(
        model=model,
        data_module=data_module,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        save_dir=args.save_dir,
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, patience=args.patience)
    
    print("\n✓ Training completed!")


if __name__ == '__main__':
    main()
