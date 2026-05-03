"""
Combined Severity DataModule
============================
DataModule for combined DR severity datasets with proper train/val/test splits.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

from .datasets.combined_severity import CombinedSeverityDataset


class CombinedSeverityDataModule:
    """
    DataModule for combined DR severity classification.
    """

    def __init__(self,
                 datasets=['eyepacs', 'aptos', 'dr_unified_v2'],
                 batch_size=32,
                 image_size=224,
                 num_workers=4,
                 balanced_sampling=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.balanced_sampling = balanced_sampling

        # Transforms will be set up in setup()
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Setup datasets and transforms."""
        from .transforms import get_severity_transforms

        self.train_transform, self.val_transform, self.test_transform = get_severity_transforms(
            image_size=self.image_size
        )

        self.train_dataset = CombinedSeverityDataset(
            datasets=self.datasets,
            split='train',
            transform=self.train_transform
        )

        self.val_dataset = CombinedSeverityDataset(
            datasets=self.datasets,
            split='val',
            transform=self.val_transform
        )

        self.test_dataset = CombinedSeverityDataset(
            datasets=self.datasets,
            split='test',
            transform=self.test_transform
        )

    def train_dataloader(self):
        """Training dataloader with optional balanced sampling."""
        if self.balanced_sampling:
            # Calculate class weights for balanced sampling
            labels = [sample['severity'] for sample in self.train_dataset.samples]
            class_counts = Counter(labels)
            total_samples = len(labels)

            # Calculate weights: inverse frequency
            class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in labels]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_severity_distribution(self):
        """Get severity class distribution for logging."""
        if hasattr(self, 'train_dataset'):
            labels = [sample['severity'] for sample in self.train_dataset.samples]
            return Counter(labels)
        return None