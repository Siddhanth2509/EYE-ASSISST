#!/usr/bin/env python3
"""
Phase 3: Multi-Disease Detection Training Script

Trains a multi-label classifier to detect multiple eye diseases:
- Diabetic Retinopathy (DR)
- Glaucoma
- Age-Related Macular Degeneration (AMD)
- Cataracts
- Hypertensive Retinopathy
- Myopic Macular Degeneration

Usage:
    python phase3_multi_disease/train.py --data_root Dataset/ --epochs 50 --model resnet50
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# MULTI-LABEL DATASET
# ============================================================================

class MultiDiseaseDataset(Dataset):
    """
    Multi-label fundus dataset.
    
    Expected CSV format:
    image_path,dr,glaucoma,amd,cataract,hypertensive,myopic
    path/to/img1.jpg,1,0,0,0,0,0
    path/to/img2.jpg,1,1,0,0,0,0
    """
    
    def __init__(self, csv_path, data_root, transform=None):
        """
        Args:
            csv_path: Path to CSV with labels
            data_root: Root directory for images
            transform: Image transformations
        """
        import pandas as pd
        
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Load CSV
        df = pd.read_csv(csv_path)
        self.image_paths = df['image_path'].values
        
        # Extract labels (all columns except image_path)
        label_cols = ['dr', 'glaucoma', 'amd', 'cataract', 'hypertensive', 'myopic']
        self.labels = df[label_cols].values.astype(np.float32)
        
        print(f"Loaded {len(self)} samples with {len(label_cols)} disease labels")
        
        # Print class distribution
        for i, col in enumerate(label_cols):
            pos_count = self.labels[:, i].sum()
            print(f"  {col}: {int(pos_count)}/{len(self)} ({100*pos_count/len(self):.1f}%)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.data_root / self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, labels


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiDiseaseModel(nn.Module):
    """
    Multi-label classification model for eye diseases.
    Uses a shared backbone with separate disease-specific heads.
    """
    
    def __init__(self, backbone='resnet50', num_diseases=6, pretrained=True):
        super().__init__()
        
        self.num_diseases = num_diseases
        
        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = base_model.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.flatten = nn.Flatten()
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_diseases)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.flatten(features)
        logits = self.classifier(features)
        return logits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_transforms(image_size=224, mode='train'):
    """Get data augmentation transforms."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def compute_metrics(preds, targets, threshold=0.5):
    """Compute evaluation metrics for multi-label classification."""
    preds_binary = (preds >= threshold).astype(int)
    
    # Per-class metrics
    f1_per_class = f1_score(targets, preds_binary, average=None, zero_division=0)
    precision_per_class = precision_score(targets, preds_binary, average=None, zero_division=0)
    recall_per_class = recall_score(targets, preds_binary, average=None, zero_division=0)
    
    # Overall metrics
    f1_micro = f1_score(targets, preds_binary, average='micro', zero_division=0)
    f1_macro = f1_score(targets, preds_binary, average='macro', zero_division=0)
    
    # AUC-ROC (requires probability predictions)
    try:
        auc_per_class = []
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) > 1:  # Need both classes
                auc_per_class.append(roc_auc_score(targets[:, i], preds[:, i]))
            else:
                auc_per_class.append(np.nan)
        auc_macro = np.nanmean(auc_per_class)
    except:
        auc_per_class = [np.nan] * targets.shape[1]
        auc_macro = np.nan
    
    return {
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'auc_per_class': auc_per_class,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    
    return avg_loss, metrics


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train multi-disease classifier')
    
    # Data
    parser.add_argument('--data_root', type=str, default='Dataset/',
                        help='Root directory for images')
    parser.add_argument('--train_csv', type=str, default='phase3_multi_disease/data/train.csv',
                        help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default='phase3_multi_disease/data/val.csv',
                        help='Path to validation CSV')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0'],
                        help='Backbone architecture')
    parser.add_argument('--num_diseases', type=int, default=6,
                        help='Number of disease classes')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Train from scratch without pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='phase3_multi_disease/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(args.save_dir) / f'multidisease_{args.model}_{timestamp}'
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_path / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Configuration saved to {save_path / 'config.json'}")
    print(f"Device: {args.device}")
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = MultiDiseaseDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        transform=get_transforms(args.image_size, mode='train')
    )
    
    val_dataset = MultiDiseaseDataset(
        csv_path=args.val_csv,
        data_root=args.data_root,
        transform=get_transforms(args.image_size, mode='val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = MultiDiseaseModel(
        backbone=args.model,
        num_diseases=args.num_diseases,
        pretrained=not args.no_pretrained
    )
    model = model.to(args.device)
    print(f"Model: {args.model} with {args.num_diseases} disease outputs")
    
    # Loss and optimizer
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n=== Starting Training ===")
    best_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'val_auc_macro': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train F1 (micro/macro): {train_metrics['f1_micro']:.4f} / {train_metrics['f1_macro']:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, args.device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val F1 (micro/macro): {val_metrics['f1_micro']:.4f} / {val_metrics['f1_macro']:.4f}")
        print(f"Val AUC (macro): {val_metrics['auc_macro']:.4f}")
        
        # Per-class metrics
        disease_names = ['DR', 'Glaucoma', 'AMD', 'Cataract', 'Hypertensive', 'Myopic']
        print("\nPer-class F1 scores:")
        for name, f1 in zip(disease_names, val_metrics['f1_per_class']):
            print(f"  {name:15s}: {f1:.4f}")
        
        # Update scheduler
        scheduler.step(val_metrics['f1_macro'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_auc_macro'].append(val_metrics['auc_macro'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'config': vars(args)
            }
            
            torch.save(checkpoint, save_path / 'best_model.pt')
            print(f"✓ Saved best model (F1 macro: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Save history
        with open(save_path / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best F1 (macro): {best_f1:.4f}")
    print(f"Model saved to: {save_path / 'best_model.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
