#!/usr/bin/env python3
"""
Evaluation script for binary DR classification.

Usage:
    python evaluate.py --checkpoint models/binary_dr/best.pt --data_root Data/splits/fundus
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models

from src.data.datamodule import FundusDataModule
from src.training.evaluate import BinaryEvaluator


def get_model(model_name: str = 'resnet18'):
    """
    Get model architecture (without pretrained weights for evaluation).
    
    Args:
        model_name: Model name ('resnet18', 'resnet50', 'efficientnet_b0', etc.)
    
    Returns:
        Model with modified final layer for binary classification
    """
    import torchvision
    # Check torchvision version for compatibility
    tv_version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    
    # Use weights parameter for torchvision >= 0.13.0, pretrained for older versions
    use_weights = tv_version >= (0, 13)
    
    if model_name == 'resnet18':
        if use_weights:
            model = models.resnet18(weights=None)
        else:
            model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'resnet50':
        if use_weights:
            model = models.resnet50(weights=None)
        else:
            model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'efficientnet_b0':
        if use_weights:
            model = models.efficientnet_b0(weights=None)
        else:
            model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate binary DR classifier')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='Data/splits/fundus',
        help='Path to data root'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50', 'efficientnet_b0'],
        help='Model architecture (must match training)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Input image size (must match training)'
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
        '--split',
        type=str,
        default='test',
        choices=['test', 'val'],
        help='Split to evaluate on (test=APTOS external, val=EyePACS validation)'
    )
    
    args = parser.parse_args()
    
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
    model = get_model(model_name=args.model)
    
    # Setup evaluator
    evaluator = BinaryEvaluator(
        model=model,
        data_module=data_module,
        checkpoint_path=args.checkpoint,
    )
    
    # Evaluate
    metrics, report, cm = evaluator.evaluate(split=args.split)
    
    # Save results
    evaluator.save_results(metrics, split=args.split)
    
    print("\n✓ Evaluation completed!")


if __name__ == '__main__':
    main()
