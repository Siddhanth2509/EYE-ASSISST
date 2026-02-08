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

        # Validate paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not Path(labels_csv).exists():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        # Load CSV
        df = pd.read_csv(labels_csv)

        # Expect columns: image, level
        if "image" not in df.columns or "level" not in df.columns:
            raise ValueError(
                f"CSV must contain columns: ['image', 'level']. "
                f"Found columns: {list(df.columns)}"
            )

        # Build mapping: image_id -> severity level
        self.label_map = {
            str(row["image"]): int(row["level"])
            for _, row in df.iterrows()
        }

        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'}

        # Collect image paths that have labels
        # Use rglob to recursively scan subdirectories (DR/, NORMAL/, etc.)
        self.samples = []
        for img_path in self.images_dir.rglob("*"):
            # Only process files with valid image extensions
            if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                image_id = img_path.stem  # filename without extension
                if image_id in self.label_map:
                    self.samples.append((img_path, self.label_map[image_id]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matching images found in {images_dir}. "
                f"Checked {len(list(self.images_dir.iterdir()))} files, "
                f"labels available for {len(self.label_map)} images."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, severity = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")

        if self.transform:
            image = self.transform(image)

        severity_label = torch.tensor(severity, dtype=torch.long)
        dr_label = torch.tensor(1 if severity > 0 else 0, dtype=torch.long)

        return image, dr_label, severity_label

    def get_severity_counts(self):
        """
        Get count of samples for each severity level.
        
        Returns:
            dict: Severity level -> count mapping
        """
        from collections import Counter
        severities = [severity for _, severity in self.samples]
        return dict(Counter(severities))
