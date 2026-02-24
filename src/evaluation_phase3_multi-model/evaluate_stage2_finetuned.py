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
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
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

    CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    print("\nPer-Class Accuracy:")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for name, acc_val in zip(CLASS_NAMES, per_class_acc):
        print(f"  {name:15s}: {acc_val:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()

