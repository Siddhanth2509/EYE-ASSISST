# src/data/eyepacs_severity_datamodule.py

import os
import torch
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.data.datasets.eyepacs_severity import EyePACSSeverityDataset


class EyePACSSeverityDataModule:
    """
    DataModule for DR severity training (Stage-2).
    Uses EyePACS only.
    """

    def __init__(
        self,
        data_root: str = "Data/splits/fundus",
        labels_csv: str = "Data/labels/eyepacs_trainLabels.csv",
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0 if os.name == "nt" else 4,
        pin_memory: bool = True,
    ):
        self.data_root = Path(data_root)
        self.labels_csv = Path(labels_csv)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.transform_train = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.transform_eval = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def setup(self):
        """
        Initialize datasets from directory structure.
        
        Note: EyePACS only has train/val splits for Stage-2.
        Test evaluation happens on APTOS in later stages.
        
        Prevents accidental double-loading if called multiple times.
        """
        if hasattr(self, "train_dataset"):
            return

        self.train_dataset = EyePACSSeverityDataset(
            images_dir=self.data_root / "eyepacs" / "train",
            labels_csv=self.labels_csv,
            transform=self.transform_train,
        )

        self.val_dataset = EyePACSSeverityDataset(
            images_dir=self.data_root / "eyepacs" / "val",
            labels_csv=self.labels_csv,
            transform=self.transform_eval,
        )

    def _get_train_labels(self):
        """Extract severity labels from training dataset samples list (fast, no I/O)."""
        return [severity for _, severity in self.train_dataset.samples]

    def get_severity_distribution(self):
        """
        Get severity level distribution in the training set.
        
        Returns:
            dict: Severity level counts (0-4)
        """
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup() before get_severity_distribution()."
            )
        return dict(Counter(self._get_train_labels()))

    def get_class_weights(self, num_classes: int = 5):
        """
        Calculate inverse-frequency class weights for imbalanced severity levels.
        
        Args:
            num_classes: Number of severity classes (default: 5 for 0-4)
        
        Returns:
            torch.Tensor: Weight for each severity class (shape: [num_classes])
        
        Note: setup() must be called before this method.
        """
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup() before get_class_weights()."
            )
        
        labels = self._get_train_labels()
        class_counts = Counter(labels)
        total = len(labels)
        
        # Inverse frequency: rare classes get higher weight
        weights = torch.zeros(num_classes)
        for cls_idx, count in class_counts.items():
            weights[cls_idx] = total / (num_classes * count)
        
        return weights

    def _build_weighted_sampler(self):
        """
        Build a WeightedRandomSampler that oversamples minority classes.
        
        Each sample gets weight = 1 / count(its_class), so all classes
        are equally likely to appear in a batch.
        """
        labels = self._get_train_labels()
        class_counts = Counter(labels)
        
        # Per-sample weight = inverse of its class frequency
        sample_weights = torch.tensor(
            [1.0 / class_counts[label] for label in labels],
            dtype=torch.float64,
        )
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def train_dataloader(self, balanced: bool = True):
        """
        Get training DataLoader.
        
        Args:
            balanced: If True, use WeightedRandomSampler for class-balanced
                      batches. If False, use plain shuffle (original behavior).
        """
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup() before train_dataloader()."
            )
        
        if balanced:
            sampler = self._build_weighted_sampler()
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,           # sampler handles ordering — no shuffle
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            raise RuntimeError(
                "val_dataset not initialized. Call setup() before val_dataloader()."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
