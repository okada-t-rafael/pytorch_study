"""
Contains various utility functions for PyTorch model training and saving.
"""
import logging
import torch
from pathlib import Path


def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str,
        ) -> bool:
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either
            ".pth" or ".pt" as the filename extension.

    Returns:
        A boolean indicating whether the save operation was executed without
        errors.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    if not model_name.endswith((".pth", ".pt")):
        logging.error("Model's name should end with '.pt' or '.pth'.")
        return False
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    try:
        torch.save(obj=model.state_dict(), f=model_save_path)
    except Exception as e:
        logging.erro(f"Error while saving model: {e}")
        return False

    return True
