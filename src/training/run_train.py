# src/training/run_train.py

import torch

from src.data.datamodule import FundusDataModule
from src.models.cnn_backbone import create_backbone
from src.training.train_binary import BinaryTrainer
from src.utils.seed import set_seed


def main():
    # Reproducibility
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize DataModule (FROZEN splits)
    data_module = FundusDataModule(
        data_root="Data/splits/fundus",
        image_size=224,
        batch_size=32,
        num_workers=4,
    )
    data_module.setup()

    # Initialize model (Stage-1 locked ResNet18)
    model = create_backbone(
        backbone="resnet18",
        pretrained=True,
    )

    # Initialize trainer
    trainer = BinaryTrainer(
        model=model,
        data_module=data_module,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
        use_class_weights=True,
        save_dir="models/binary_dr",
    )

    # Start training
    trainer.train(
        num_epochs=50,
        patience=10,
    )


if __name__ == "__main__":
    main()
