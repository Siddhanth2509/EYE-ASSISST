"""
E4 — EfficientNet-B3 @ 384px Training Script
----------------------------------------------
Purpose:
    Train DR Severity Head on EfficientNet-B3 backbone at 384px resolution.
    Fresh training from ImageNet-pretrained weights (no Stage-1 checkpoint).

Strategy:
    Phase 1 (epochs 1-10): Freeze backbone, train severity head only
    Phase 2 (epochs 11-25): Unfreeze last 2 EfficientNet blocks + head

Why EfficientNet-B3:
    - 1536-dim features vs ResNet18's 512 → 3x richer representation
    - 384px input captures subtle microaneurysms in Mild DR
    - Compound scaling (depth + width + resolution) is optimal for medical imaging

Primary Metric:
    Quadratic Weighted Kappa (QWK)

Target: QWK ≥ 0.70 (from ResNet18 ceiling of 0.6456)

Author: Siddhanth Sharma
Phase: 3 (E4 — Architecture Upgrade)
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import numpy as np

from src.models.multi_task_models import MultiTaskModel
from src.data.eyepacs_severity_datamodule import EyePACSSeverityDataModule
from src.losses.severity_loss import SeverityLoss
from src.utils.checkpoints import save_checkpoint
from src.utils.seed import set_seed


# =========================
# CONFIG
# =========================
BACKBONE = "efficientnet_b3"
IMAGE_SIZE = 384
BATCH_SIZE = 16           # 384px on T4 (16GB): ~16 comfortably
NUM_EPOCHS = 25
UNFREEZE_EPOCH = 10       # Unfreeze top blocks after this epoch
LR_HEAD = 1e-4            # Severity head learning rate
LR_BACKBONE_FT = 1e-5    # Backbone fine-tune LR (10x lower)
WEIGHT_DECAY = 1e-4
SEED = 42

SAVE_DIR = "models/stage2_efficientnet"
HISTORY_PATH = os.path.join(SAVE_DIR, "training_history.json")

# Set num_workers based on environment
NUM_WORKERS = 4 if torch.cuda.is_available() else 0


# =========================
# Freeze / Unfreeze Utilities
# =========================
def freeze_backbone(model):
    """Freeze entire backbone."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()


def freeze_binary_head(model):
    """Freeze binary head (not used in Stage-2 but keep frozen)."""
    for param in model.dr_binary_head.parameters():
        param.requires_grad = False
    model.dr_binary_head.eval()


def unfreeze_top_blocks(model):
    """
    Unfreeze the last 2 EfficientNet feature blocks for fine-tuning.
    
    EfficientNet model.features is a Sequential of 9 blocks (indices 0-8).
    We unfreeze blocks 7 and 8 (the deepest, most task-specific layers).
    """
    blocks_to_unfreeze = model.backbone.get_finetune_layers()
    
    total_unfrozen = 0
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True
            total_unfrozen += param.numel()
    
    head_params = sum(p.numel() for p in model.dr_severity_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Backbone Unfreeze ---")
    print(f"Unfrozen backbone params: {total_unfrozen:,}")
    print(f"Severity head params:     {head_params:,}")
    print(f"Total trainable:          {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
    
    return blocks_to_unfreeze


def count_parameters(model):
    """Print parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
    return total, trainable


# =========================
# Training Loop
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num):
    model.dr_severity_head.train()
    # Backbone may be in train or eval mode depending on freeze state
    
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train E{epoch_num}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images, dr_labels, severity_labels = batch

        images = images.to(device)
        severity_labels = severity_labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        severity_logits = outputs["dr_severity"]

        loss = criterion(severity_logits, severity_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{running_loss/num_batches:.4f}"})

    return running_loss / num_batches


# =========================
# Validation Loop
# =========================
def validate(model, loader, criterion, device):
    model.eval()

    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False, dynamic_ncols=True):
            images, dr_labels, severity_labels = batch

            images = images.to(device)
            severity_labels = severity_labels.to(device)

            outputs = model(images)
            severity_logits = outputs["dr_severity"]

            loss = criterion(severity_logits, severity_labels)
            val_loss += loss.item()

            preds = torch.argmax(severity_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(severity_labels.cpu().numpy())

    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return val_loss / len(loader), qwk, accuracy


# =========================
# Main
# =========================
def main():
    print("=" * 60)
    print("E4 — EfficientNet-B3 @ 384px Severity Training")
    print("=" * 60)

    # 1️⃣ Reproducibility
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 2️⃣ DataModule (384px resolution)
    datamodule = EyePACSSeverityDataModule(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader(balanced=False)
    val_loader = datamodule.val_dataloader()

    dist = datamodule.get_severity_distribution()
    print(f"Training class distribution: {dist}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3️⃣ Model — fresh EfficientNet-B3 with ImageNet pretrained backbone
    model = MultiTaskModel(backbone=BACKBONE, backbone_pretrained=True)
    model.to(device)
    
    print(f"\nBackbone: {BACKBONE}")
    print(f"Feature dim: {model.backbone.feature_dim}")
    print(f"Image size: {IMAGE_SIZE}px")
    count_parameters(model)

    # 4️⃣ Phase 1: Freeze backbone + binary head, train severity head only
    freeze_backbone(model)
    freeze_binary_head(model)
    print("\nPhase 1: Backbone frozen — training severity head only.")
    count_parameters(model)

    # 5️⃣ Loss (plain CE)
    criterion = SeverityLoss()

    # 6️⃣ Optimizer — initially only severity head
    optimizer = torch.optim.AdamW(
        model.dr_severity_head.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_qwk = -1.0
    history = []

    # 7️⃣ Training Loop
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # --- Phase transition: unfreeze top blocks ---
        if epoch == UNFREEZE_EPOCH:
            print(f"\n{'='*60}")
            print(f"Phase 2: Unfreezing top EfficientNet blocks (epoch {epoch+1})")
            print(f"{'='*60}")
            
            blocks = unfreeze_top_blocks(model)
            
            # Rebuild optimizer with discriminative LR
            optimizer = torch.optim.AdamW([
                {"params": model.dr_severity_head.parameters(), "lr": LR_HEAD * 0.5},
                *[{"params": block.parameters(), "lr": LR_BACKBONE_FT} for block in blocks],
            ], weight_decay=WEIGHT_DECAY)
            
            # Reset scheduler for phase 2
            remaining = NUM_EPOCHS - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-6
            )
            
            # Set backbone blocks to train mode
            for block in blocks:
                block.train()

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1
        )

        val_loss, val_qwk, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val QWK:    {val_qwk:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print(f"LR:         {current_lr:.2e}")
        print(f"Time:       {epoch_time:.1f}s")

        # Track history
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_qwk": round(val_qwk, 4),
            "val_acc": round(val_acc, 4),
            "lr": current_lr,
            "time_sec": round(epoch_time, 1),
            "phase": 1 if epoch < UNFREEZE_EPOCH else 2,
        })

        # Save best model (based on QWK)
        if val_qwk > best_qwk:
            best_qwk = val_qwk

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=os.path.join(SAVE_DIR, "best.pt"),
                backbone=BACKBONE,
                image_size=IMAGE_SIZE,
                val_qwk=val_qwk,
                val_acc=val_acc,
            )

            print(f"✅ New best model saved (QWK={val_qwk:.4f})")

        # Save latest checkpoint every epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=os.path.join(SAVE_DIR, "latest.pt"),
            backbone=BACKBONE,
            image_size=IMAGE_SIZE,
            val_qwk=val_qwk,
            val_acc=val_acc,
        )

        # Save history after each epoch
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"E4 Training Complete")
    print(f"{'='*60}")
    print(f"Best QWK:   {best_qwk:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"History saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
