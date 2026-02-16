# src/utils/checkpoints.py

import os
import torch


def save_checkpoint(model, optimizer, epoch, path, **extra):
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch number.
        path: File path to save the checkpoint.
        **extra: Any additional key-value pairs to store.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: (Optional) Optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        dict: The full checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
