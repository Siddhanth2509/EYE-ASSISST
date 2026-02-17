# src/data/eyepacs_severity_datamodule.py

import os
from pathlib import Path
from torch.utils.data import DataLoader
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
        
        from collections import Counter
        severity_labels = [self.train_dataset[i][2].item() for i in range(len(self.train_dataset))]
        return dict(Counter(severity_labels))

    def get_class_weights(self, num_classes: int = 5):
        """
        Calculate class weights for imbalanced severity levels.
        
        Args:
            num_classes: Number of severity classes (default: 5 for 0-4)
        
        Returns:
            torch.Tensor: Weight for each severity class
        
        Note: setup() must be called before this method.
        """
        from collections import Counter
        import torch
        
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup() before get_class_weights()."
            )
        
        # Get severity labels from training set
        severity_labels = [self.train_dataset[i][2].item() for i in range(len(self.train_dataset))]
        class_counts = Counter(severity_labels)
        
        total = len(severity_labels)
        
        # Calculate weights: inverse frequency
        weights = torch.zeros(num_classes)
        for idx, count in class_counts.items():
            weights[idx] = total / (num_classes * count)
        
        return weights

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup() before train_dataloader()."
            )
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
