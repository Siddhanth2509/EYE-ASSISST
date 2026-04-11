#!/usr/bin/env python3
"""
Phase 3 - Threshold Calibration Script

Finds the optimal decision threshold per disease class by scanning
thresholds from 0.1 to 0.9 and picking the one that maximises F1.

Usage:
  python phase3_multi_disease/calibrate_thresholds.py \
    --checkpoint phase3_multi_disease/checkpoints/<folder>/best_model.pt \
    --val_csv    phase3_multi_disease/data/val_unified_v4.csv \
    --data_root  .
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DISEASE_NAMES = ['DR', 'Glaucoma', 'AMD', 'Cataract', 'Hypertensive', 'Myopic']
LABEL_COLS    = ['dr', 'glaucoma', 'amd', 'cataract', 'hypertensive', 'myopic']


# ── Dataset ──────────────────────────────────────────────────────────────────
class MultiDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_root, transform=None):
        df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.image_paths = df['image_path'].values
        self.labels = df[LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.data_root / self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx])


# ── Model ─────────────────────────────────────────────────────────────────────
class MultiDiseaseModel(nn.Module):
    def __init__(self, backbone='resnet50', num_diseases=6):
        super().__init__()
        base = models.resnet50(weights=None)
        feat_dim = base.fc.in_features
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_diseases)
        )

    def forward(self, x):
        return self.classifier(self.flatten(self.backbone(x)))


# ── Inference ─────────────────────────────────────────────────────────────────
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating'):
            logits = model(imgs.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_csv',    type=str, required=True)
    parser.add_argument('--data_root',  type=str, default='.')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get('config', {})
    backbone = config.get('model', 'resnet50')

    model = MultiDiseaseModel(backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Best F1 during training: {checkpoint.get('best_f1', '?'):.4f}")

    # Val dataset
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_ds = MultiDiseaseDataset(args.val_csv, args.data_root, val_tf)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Get raw probabilities
    print("\nRunning inference on validation set...")
    probs, labels = get_predictions(model, val_loader, args.device)

    print(f"\nPrediction distribution (sigmoid outputs):")
    for i, name in enumerate(DISEASE_NAMES):
        print(f"  {name:13s}: mean={probs[:,i].mean():.3f}  "
              f"min={probs[:,i].min():.3f}  max={probs[:,i].max():.3f}")

    # AUC per class (threshold-independent)
    print("\n=== AUC per class (threshold-independent) ===")
    for i, name in enumerate(DISEASE_NAMES):
        if len(np.unique(labels[:,i])) > 1:
            auc = roc_auc_score(labels[:,i], probs[:,i])
            print(f"  {name:13s}: AUC = {auc:.4f}")
        else:
            print(f"  {name:13s}: AUC = N/A (single class in val)")

    # Scan thresholds per class
    thresholds = np.arange(0.10, 0.90, 0.05)
    print("\n=== Optimal Threshold Search ===")
    best_thresholds = {}

    for i, name in enumerate(DISEASE_NAMES):
        best_f1, best_t = 0.0, 0.5
        for t in thresholds:
            preds_bin = (probs[:,i] >= t).astype(int)
            f1 = f1_score(labels[:,i], preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[name.lower()] = float(round(best_t, 2))
        print(f"  {name:13s}: best threshold = {best_t:.2f}  ->  F1 = {best_f1:.4f}")

    # Show final metrics at optimal thresholds
    print("\n=== Final Performance at Optimal Thresholds ===")
    per_class_f1 = []
    for i, name in enumerate(DISEASE_NAMES):
        t = best_thresholds[name.lower()]
        preds_bin = (probs[:,i] >= t).astype(int)
        f1 = f1_score(labels[:,i], preds_bin, zero_division=0)
        per_class_f1.append(f1)
        tag = "GOOD" if f1 > 0.5 else ("OK" if f1 > 0.3 else "LOW")
        print(f"  {name:13s}: F1 = {f1:.4f}  [{tag}]")

    macro_f1 = np.mean(per_class_f1)
    print(f"\n  Macro F1 @ optimal thresholds: {macro_f1:.4f}")
    print(f"  (vs  F1 @ 0.5 fixed threshold: {checkpoint.get('best_f1', '?'):.4f})")

    # Save threshold config for backend
    output = {
        "model_backbone": backbone,
        "checkpoint_epoch": checkpoint.get('epoch'),
        "thresholds": best_thresholds,
        "macro_f1_at_optimal_threshold": round(float(macro_f1), 4),
        "macro_f1_at_0_5_threshold": round(float(checkpoint.get('best_f1', 0)), 4),
    }

    out_path = Path(args.checkpoint).parent / "threshold_config.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAVED] Threshold config -> {out_path}")
    print("\nBackend integration snippet:")
    print("  thresholds = {")
    for k, v in best_thresholds.items():
        print(f'    "{k}": {v},')
    print("  }")


if __name__ == '__main__':
    main()
