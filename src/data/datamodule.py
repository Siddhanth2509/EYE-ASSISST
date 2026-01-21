# src/data/datamodule.py

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class FundusDataModule:
    """
    DataModule for fundus image classification.

    - Uses FROZEN splits from Phase 1
    - Supports EyePACS (train/val) and APTOS (test)
    - Includes proper normalization for medical images
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # ImageNet normalization (standard for pretrained models)
        # Can be changed to fundus-specific if needed
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.transform_train = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),  # Slightly larger for cropping
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
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
        Initialize datasets from frozen directory structure.
        
        Prevents accidental double-loading if called multiple times.
        Common pattern in Lightning-style DataModules.
        """
        # Guard: prevent double-loading if already initialized
        if hasattr(self, "train_dataset"):
            return
        
        # Data structure: Data/splits/fundus/eyepacs/{train,val}/ and aptos/test/
        self.train_dataset = datasets.ImageFolder(
            self.data_root / "eyepacs" / "train",
            transform=self.transform_train,
        )

        self.val_dataset = datasets.ImageFolder(
            self.data_root / "eyepacs" / "val",
            transform=self.transform_eval,
        )

        self.test_dataset = datasets.ImageFolder(
            self.data_root / "aptos" / "test",
            transform=self.transform_eval,
        )
        
        # CRITICAL: Verify class order is consistent across all splits
        # ImageFolder assigns labels alphabetically, so this ensures:
        # - DR = 0, NORMAL = 1 (or vice versa, but FIXED)
        # - Metrics remain meaningful
        # - Grad-CAM interpretations are correct
        # - Clinical interpretation is valid
        expected_classes = ["DR", "NORMAL"]
        assert self.train_dataset.classes == expected_classes, (
            f"Unexpected train class order: {self.train_dataset.classes}. "
            f"Expected: {expected_classes}"
        )
        assert self.val_dataset.classes == expected_classes, (
            f"Unexpected val class order: {self.val_dataset.classes}. "
            f"Expected: {expected_classes}"
        )
        assert self.test_dataset.classes == expected_classes, (
            f"Unexpected test class order: {self.test_dataset.classes}. "
            f"Expected: {expected_classes}"
        )

    def get_class_weights(self):
        """
        Calculate class weights for imbalanced dataset.
        
        Note: setup() must be called before this method.
        """
        from collections import Counter
        import torch
        
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError(
                "train_dataset not initialized. Call setup() before get_class_weights()."
            )
        
        # Count samples per class in training set
        targets = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        class_counts = Counter(targets)
        
        total = len(targets)
        n_classes = len(class_counts)
        
        # Calculate weights: inverse frequency
        weights = torch.zeros(n_classes)
        for idx, count in class_counts.items():
            weights[idx] = total / (n_classes * count)
        
        return weights

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
