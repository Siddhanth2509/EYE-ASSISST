"""
Stage-2 Fine-Tuned Model Evaluation
------------------------------------
Evaluates:
    - Accuracy
    - Per-class accuracy
    - Confusion Matrix
    - Quadratic Weighted Kappa (QWK)

Author: Siddhanth Sharma
"""

import torch
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score,
    classification_report
)

from src.models.multi_task_models import MultiTaskModel
from src.data.eyepacs_severity_datamodule import EyePACSSeverityDataModule


CHECKPOINT_PATH = "models/stage2_dr_severity/finetuned_best.pt"
BATCH_SIZE = 32
IMAGE_SIZE = 224


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    datamodule = EyePACSSeverityDataModule(
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    datamodule.setup()

    val_loader = datamodule.val_dataloader()

    # Model
    model = MultiTaskModel(backbone="resnet50", backbone_pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, _, severity_labels in val_loader:

            images = images.to(device)
            severity_labels = severity_labels.to(device)

            outputs = model(images)
            logits = outputs["dr_severity"]

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(severity_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== Stage-2 Fine-Tuned Evaluation =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"QWK:      {qwk:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    main()