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
NUM_EPOCHS = 25
LR = 5e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 224
SEED = 42


# =========================
# Freeze Utility
# =========================
def freeze_backbone_and_binary(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.dr_binary_head.parameters():
        param.requires_grad = False

    model.backbone.eval()
    model.dr_binary_head.eval()
    model.dr_severity_head.train()


# =========================
# Training Loop
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
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
        for batch in tqdm(loader, desc="Validation"):
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

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # 3️⃣ Model
    model = MultiTaskModel(backbone="resnet50")
    checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("Stage-1 weights loaded successfully.")

    # 4️⃣ Freeze
    freeze_backbone_and_binary(model)
    print("Backbone and Binary head frozen.")

    # 5️⃣ Loss
    criterion = SeverityLoss()

    # 6️⃣ Optimizer (ONLY severity head params)
    optimizer = torch.optim.Adam(
        model.dr_severity_head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_qwk = -1.0

    # 7️⃣ Training Loop
    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

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
