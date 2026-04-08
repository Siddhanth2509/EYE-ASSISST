"""
Stage-2 Fine-Tuning Script
---------------------------
Purpose:
    Fine-tune Stage-2 severity model by:
        - Loading best Stage-2 checkpoint
        - Unfreezing backbone (BN layers stay frozen)
        - Keeping binary head frozen
        - Using class-weighted loss to handle imbalance
        - Cosine LR schedule + gradient clipping

Strategy:
    Backbone LR  : 1e-5
    Severity LR  : 5e-5
    Epochs       : 15
    Metric       : Quadratic Weighted Kappa (QWK)

Author: Siddhanth Sharma
Phase: 3 (Stage-2 Fine-Tune)
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

from src.models.multi_task_models import MultiTaskModel
from src.data.eyepacs_severity_datamodule import EyePACSSeverityDataModule
from src.utils.checkpoints import save_checkpoint
from src.utils.seed import set_seed


# =========================
# CONFIG
# =========================
STAGE2_CHECKPOINT = "models/stage2_dr_severity/best.pt"
SAVE_DIR = "models/stage2_dr_severity"
NUM_EPOCHS = 15

BACKBONE_LR = 1e-5
HEAD_LR = 5e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

BATCH_SIZE = 32
IMAGE_SIZE = 224
SEED = 42


# =========================
# Freeze BN Utility
# =========================
def freeze_bn(module):
    """Recursively set all BatchNorm layers to eval mode."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


# =========================
# Training Loop
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    model.train()

    # Keep backbone BN frozen so running stats are not corrupted
    freeze_bn(model.backbone)

    # Keep binary head fully in eval mode (frozen)
    model.dr_binary_head.eval()

    running_loss = 0.0

    for images, _, severity_labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        severity_labels = severity_labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        severity_logits = outputs["dr_severity"]

        loss = criterion(severity_logits, severity_labels)
        loss.backward()

        # Clip gradients to prevent destabilising backbone
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
        for images, _, severity_labels in tqdm(loader, desc="Validation", leave=False):
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

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────
    datamodule = EyePACSSeverityDataModule(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # ── Class weights (handle EyePACS imbalance) ─────────
    class_weights = datamodule.get_class_weights(num_classes=5).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # ── Model ─────────────────────────────────────────────
    model = MultiTaskModel(backbone="resnet50", backbone_pretrained=False)
    checkpoint = torch.load(STAGE2_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    prev_epoch = checkpoint.get("epoch", -1)
    print(f"Loaded Stage-2 best model (from epoch {prev_epoch + 1}).")

    # ── Freeze / Unfreeze ─────────────────────────────────
    # Unfreeze backbone conv weights (BN stays eval via freeze_bn)
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Keep binary head completely frozen
    for param in model.dr_binary_head.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    print("Backbone unfrozen (BN frozen). Binary head frozen.")

    # ── Loss (class-weighted CE) ──────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer (discriminative LR) ─────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.dr_severity_head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    # ── LR Scheduler (cosine annealing) ──────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_qwk = -1.0
    patience_counter = 0
    PATIENCE = 5

    # ── Training Loop ─────────────────────────────────────
    for epoch in range(NUM_EPOCHS):

        print(f"\nFine-Tune Epoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, MAX_GRAD_NORM
        )

        val_loss, val_qwk = validate(
            model, val_loader, criterion, device
        )

        current_bb_lr = optimizer.param_groups[0]["lr"]
        current_hd_lr = optimizer.param_groups[1]["lr"]

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val QWK:    {val_qwk:.4f}")
        print(f"LR (backbone): {current_bb_lr:.2e}  LR (head): {current_hd_lr:.2e}")

        scheduler.step()

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=os.path.join(SAVE_DIR, "finetuned_best.pt"),
                val_qwk=best_qwk,
                val_loss=val_loss,
            )

            print("New best fine-tuned model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nFine-tuning complete.")
    print(f"Best Fine-Tuned QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
