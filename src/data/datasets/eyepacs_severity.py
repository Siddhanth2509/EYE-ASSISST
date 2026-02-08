# src/data/datasets/eyepacs_severity.py

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class EyePACSSeverityDataset(Dataset):
    """
    EyePACS dataset with DR severity labels (0–4).

    Returns:
        image: Tensor
        dr_label: 0 (No DR) or 1 (DR)
        severity_label: 0–4 (meaningful only if dr_label == 1)
    """

    def __init__(self, images_dir: Path, labels_csv: Path, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Load CSV
        df = pd.read_csv(labels_csv)

        # Expect columns: image, level
        assert "image" in df.columns and "level" in df.columns, (
            "CSV must contain columns: ['image', 'level']"
        )

        # Build mapping: image_id -> severity level
        self.label_map = {
            str(row["image"]): int(row["level"])
            for _, row in df.iterrows()
        }

        # Collect image paths that have labels
        self.samples = []
        for img_path in self.images_dir.glob("*"):
            image_id = img_path.stem  # filename without extension
            if image_id in self.label_map:
                self.samples.append((img_path, self.label_map[image_id]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No matching images found in {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, severity = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        severity_label = torch.tensor(severity, dtype=torch.long)
        dr_label = torch.tensor(1 if severity > 0 else 0, dtype=torch.long)

        return image, dr_label, severity_label
