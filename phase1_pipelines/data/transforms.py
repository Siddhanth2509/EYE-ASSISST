"""
Data Transformations for DR Classification
==========================================
Image augmentations and preprocessing for fundus images.

Fundus-specific augmentations:
- Horizontal/vertical flip (fundus symmetry)
- Rotation (camera angle variation)
- Color jitter (lighting conditions)
- Gaussian blur (focus variation)
- Random affine (slight perspective changes)
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_severity_transforms(image_size=224, augment_strength='medium'):
    """
    Get transforms for DR severity training.

    Args:
        image_size: Target image size (square)
        augment_strength: 'light', 'medium', or 'strong'

    Returns:
        train_transform, val_transform, test_transform
    """
    
    # Augmentation parameters based on strength
    aug_params = {
        'light': {
            'rotation': 15,
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.05,
            'blur_prob': 0.1,
        },
        'medium': {
            'rotation': 30,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'blur_prob': 0.2,
        },
        'strong': {
            'rotation': 45,
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.15,
            'blur_prob': 0.3,
        }
    }
    
    params = aug_params.get(augment_strength, aug_params['medium'])

    # Training transforms with fundus-specific augmentation
    train_transform = T.Compose([
        T.Resize((image_size + 32, image_size + 32), interpolation=InterpolationMode.BILINEAR),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=params['rotation'], interpolation=InterpolationMode.BILINEAR),
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR
        ),
        T.ColorJitter(
            brightness=params['brightness'],
            contrast=params['contrast'],
            saturation=params['saturation'],
            hue=params['hue']
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=params['blur_prob']),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Small random erasing
    ])

    # Validation/Test transforms (no augmentation, center crop for consistency)
    val_test_transform = T.Compose([
        T.Resize((image_size + 16, image_size + 16), interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform, val_test_transform


def get_binary_transforms(image_size=224):
    """
    Get transforms for binary DR classification.
    """
    return get_severity_transforms(image_size, augment_strength='medium')


def get_tta_transforms(image_size=224, n_augments=5):
    """
    Get Test-Time Augmentation transforms for inference.
    Returns a list of transforms for TTA.
    """
    base_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tta_transforms = [base_transform]  # Original
    
    if n_augments >= 2:
        # Horizontal flip
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    if n_augments >= 3:
        # Vertical flip
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    if n_augments >= 4:
        # Rotation 90
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=(90, 90)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    if n_augments >= 5:
        # Rotation -90
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=(-90, -90)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    return tta_transforms
