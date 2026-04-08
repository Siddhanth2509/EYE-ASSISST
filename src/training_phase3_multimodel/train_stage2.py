"""
Stage-2 Training Script
------------------------
Purpose:
    Train DR Severity Head while freezing:
        - Backbone
        - Binary DR head

Strategy:
    1. Load Stage-1 checkpoint
    2. Freeze representation layers
    3. Train severity head only
    4. Select best model based on QWK

Primary Metric:
    Quadratic Weighted Kappa (QWK)

Author: Siddhanth Sharma
Phase: 3 (Stage-2)
"""

import os
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
# CONFIG (Stage-2 Locked)
# =========================
STAGE1_CHECKPOINT = "models/stage1_dr_binary/best.pt"
SAVE_DIR = "models/stage2_dr_severity"
NUM_EPOCHS = 10           # Continuation: 10 more epochs from E3 best
LR_HEAD = 5e-5           # Severity head learning rate
LR_BACKBONE = 5e-6       # Backbone layer4 learning rate (10x lower)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 224
SEED = 42

# Resume from E3 best checkpoint (QWK=0.6407 at epoch 12)
RESUME_CHECKPOINT = "models/stage2_dr_severity/best.pt"


# =========================
# Freeze Utility
# =========================
def freeze_backbone_and_binary(model):
    """Freeze entire backbone and binary head."""
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.dr_binary_head.parameters():
        param.requires_grad = False

    model.backbone.eval()
    model.dr_binary_head.eval()
    model.dr_severity_head.train()


def unfreeze_backbone_layer4(model):
    """
    Unfreeze only the last ResNet block (layer4) for fine-tuning.
    
    ResNet18 architecture:
        - layer1, layer2, layer3: frozen (general features)
        - layer4: unfrozen (task-specific fine-grained features)
    
    This enables the backbone to adapt its high-level representations
    for severity grading while preserving low/mid-level features.
    """
    for param in model.backbone.model.layer4.parameters():
        param.requires_grad = True
    
    # Count trainable params for logging
    layer4_params = sum(p.numel() for p in model.backbone.model.layer4.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.dr_severity_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Layer4 trainable params: {layer4_params:,}")
    print(f"Severity head params:    {head_params:,}")
    print(f"Total trainable:         {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")


# =========================
# Training Loop
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False, dynamic_ncols=True):
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

    return running_loss / len(loader)


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

    qwk = cohen_kappa_score(
        all_labels,
        all_preds,
        weights="quadratic"
    )

    return val_loss / len(loader), qwk


# =========================
# Main
# =========================
def main():

    # 1️⃣ Reproducibility
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2️⃣ DataModule
    datamodule = EyePACSSeverityDataModule(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    datamodule.setup()

    # Balanced sampling: oversample minority classes
    train_loader = datamodule.train_dataloader(balanced=False)
    val_loader = datamodule.val_dataloader()

    # Print class distribution for logging
    dist = datamodule.get_severity_distribution()
    print(f"Training class distribution: {dist}")

    # 3️⃣ Model — resume from E3 best checkpoint
    model = MultiTaskModel(backbone="resnet50")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    resume_epoch = checkpoint.get("epoch", 0)
    print(f"Resumed from E3 best checkpoint (epoch {resume_epoch + 1}, QWK=0.6407).")

    # 4️⃣ Freeze all, then selectively unfreeze layer4
    freeze_backbone_and_binary(model)
    print("Backbone and Binary head frozen.")
    
    unfreeze_backbone_layer4(model)
    print("Backbone layer4 unfrozen for fine-tuning.")

    # 5️⃣ Loss (plain CE — sampler handles class balance)
    criterion = SeverityLoss()

    # 6️⃣ Optimizer (discriminative LR: head fast, backbone slow)
    optimizer = torch.optim.Adam([
        {"params": model.dr_severity_head.parameters(), "lr": LR_HEAD},
        {"params": model.backbone.model.layer4.parameters(), "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_qwk = 0.6407  # E3 best — only save if we beat this

    # 7️⃣ Training Loop
    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] (global {resume_epoch + 1 + epoch + 1})")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_qwk = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val QWK:  {val_qwk:.4f}")

        # Save best model (based on QWK)
        if val_qwk > best_qwk:
            best_qwk = val_qwk

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=os.path.join(SAVE_DIR, "best.pt")
            )

            print("✅ New best model saved.")

    print("\nStage-2 Training Complete.")
    print(f"Best QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
