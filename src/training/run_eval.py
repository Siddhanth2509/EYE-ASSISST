# src/training/run_eval.py

import torch

from src.data.datamodule import FundusDataModule
from src.models.cnn_backbone import create_backbone
from src.training.evaluate import BinaryEvaluator
from src.utils.seed import set_seed


def main():
    # Reproducibility
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load frozen data
    data_module = FundusDataModule(
        data_root="Data/splits/fundus",
        batch_size=32,
        num_workers=4,
    )
    data_module.setup()

    # Load model architecture (same as training)
    model = create_backbone(
        backbone="resnet18",
        pretrained=False,  # weights come from checkpoint
    )

    # Load evaluator with BEST checkpoint
    evaluator = BinaryEvaluator(
        model=model,
        data_module=data_module,
        checkpoint_path="models/binary_dr/best.pt",
        device=device,
    )

    # External evaluation on APTOS
    evaluator.evaluate(split="test")


if __name__ == "__main__":
    main()
