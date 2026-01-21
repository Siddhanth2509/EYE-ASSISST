# src/data/datamodule.py

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class FundusDataModule:
    """
    DataModule for fundus image classification.

    - Uses FROZEN splits from Phase 1
    - Supports EyePACS (train/val) and APTOS (test)
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

        self.transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.transform_eval = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def setup(self):
        """Initialize datasets from frozen directory structure."""

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
