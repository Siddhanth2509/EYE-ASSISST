"""
Combined DR Severity Dataset
============================
Combines multiple DR severity datasets for improved generalization and QWK.

Supported datasets (with DR severity grades 0-4):
- APTOS (grades 0-4) - ~3,300 images
- DR Unified v2 (grades 0-4) - ~93,000 images

NOT supported (no DR severity labels):
- ODIR (multi-disease, not DR severity)
- Cataract (binary classification, not DR severity)
- EyePACS (labels not available in current setup)

Usage:
    dataset = CombinedSeverityDataset(
        datasets=['aptos', 'dr_unified_v2'],
        split='train'
    )
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class CombinedSeverityDataset(Dataset):
    """
    Combined dataset for DR severity classification across multiple sources.
    """

    def __init__(self, datasets, split='train', transform=None, base_path=None):
        """
        Args:
            datasets: List of dataset names ['aptos', 'dr_unified_v2']
            split: 'train', 'val', or 'test'
            transform: Image transformations
            base_path: Base path to Data directory
        """
        self.datasets = datasets
        self.split = split
        self.transform = transform
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent.parent.parent / "Dataset"

        # Collect all samples
        self.samples = []
        self._load_datasets()

    def _load_datasets(self):
        """Load samples from all specified datasets."""
        
        # Filter out unsupported datasets with warning
        unsupported = ['odir', 'cataract', 'eyepacs']
        
        for dataset_name in self.datasets:
            if dataset_name.lower() in unsupported:
                print(f"[SKIP] {dataset_name} - no DR severity labels (0-4) available")
                continue
            elif dataset_name == 'aptos':
                self._load_aptos()
            elif dataset_name == 'dr_unified_v2':
                self._load_dr_unified_v2()
            elif dataset_name == 'augmented_resized':
                self._load_augmented_resized()
            else:
                print(f"Warning: Unknown dataset {dataset_name}, skipping")

        print(f"[LOADED] {len(self.samples)} samples for {self.split} split")

    def _load_aptos(self):
        """Load APTOS severity dataset with proper train/val split."""
        
        # APTOS has separate train and validation CSVs
        if self.split == 'train':
            labels_path = self.base_path / "APTOS" / "train_1.csv"
            images_base = self.base_path / "APTOS" / "train_images" / "train_images"
        elif self.split == 'val':
            labels_path = self.base_path / "APTOS" / "valid.csv"
            images_base = self.base_path / "APTOS" / "val_images" / "val_images"
        elif self.split == 'test':
            labels_path = self.base_path / "APTOS" / "test.csv"
            images_base = self.base_path / "APTOS" / "test_images" / "test_images"
        else:
            print(f"Warning: Unknown split {self.split} for APTOS")
            return

        if not labels_path.exists():
            print(f"Warning: APTOS {self.split} labels not found at {labels_path}")
            return
        
        if not images_base.exists():
            # Try alternate path structure
            images_base = images_base.parent
            if not images_base.exists():
                print(f"Warning: APTOS {self.split} images not found")
                return

        df = pd.read_csv(labels_path)
        loaded = 0

        for _, row in df.iterrows():
            img_id = row['id_code']
            severity = row['diagnosis']

            # Try both .png and .jpg extensions
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = images_base / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path is None:
                # Try without nested folder
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate = images_base.parent / f"{img_id}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break

            if img_path and img_path.exists():
                self.samples.append({
                    'path': str(img_path),
                    'severity': int(severity),
                    'source': 'aptos'
                })
                loaded += 1
        
        print(f"  [APTOS {self.split}] Loaded {loaded} images")

    def _load_dr_unified_v2(self):
        """Load DR Unified v2 severity dataset."""
        images_base = self.base_path / "dr_unified_v2" / "dr_unified_v2" / self.split

        if not images_base.exists():
            print(f"Warning: DR Unified v2 {self.split} not found at {images_base}")
            return

        loaded = 0
        for severity in range(5):  # 0-4
            severity_dir = images_base / str(severity)
            if severity_dir.exists():
                for img_file in severity_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append({
                            'path': str(img_file),
                            'severity': severity,
                            'source': 'dr_unified_v2'
                        })
                        loaded += 1
        
        print(f"  [DR Unified v2 {self.split}] Loaded {loaded} images")

    def _load_augmented_resized(self):
        """Load augmented_resized_V2 severity dataset (largest dataset with 143k images)."""
        images_base = self.base_path / "augmented_resized_V2" / self.split

        if not images_base.exists():
            print(f"Warning: augmented_resized_V2 {self.split} not found at {images_base}")
            return

        loaded = 0
        for severity in range(5):  # 0-4
            severity_dir = images_base / str(severity)
            if severity_dir.exists():
                for img_file in severity_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append({
                            'path': str(img_file),
                            'severity': severity,
                            'source': 'augmented_resized'
                        })
                        loaded += 1
        
        print(f"  [Augmented Resized {self.split}] Loaded {loaded} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        severity = sample['severity']

        return image, severity
    
    def get_class_distribution(self):
        """Get class distribution for logging/debugging."""
        from collections import Counter
        labels = [s['severity'] for s in self.samples]
        return Counter(labels)
    
    def get_source_distribution(self):
        """Get distribution by data source."""
        from collections import Counter
        sources = [s['source'] for s in self.samples]
        return Counter(sources)