"""
Contains functionality for creating PyTorch DataLoader for image classification
data.
"""
import logging
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import List, Tuple


# Default number of workers for DataLoader
NUM_WORKERS = os.cpu_count()


class ErrorDataset(Dataset):
    """Dummy Dataset for creating a ZeroValue instance for DataLoaders."""
    def __len__(self):
        return 0


def create_dataloaders(
        image_path: Path,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        batch_size: int=32,
        num_workers: int=NUM_WORKERS
        ) -> Tuple[Tuple[DataLoader, DataLoader], List[str], bool]:
    """Creates training and testing DataLoaders.

    Takes in a image path directory and turns them into PyTorch Datasets and
    then into PyTorch DataLoaders.

    Args:
        image_path: Location where the train and test forlder are located.
        train_transform: Transformations to be applied to the train images.
        test_transform: Transformations to be applied to the test images.
        batch_size: Size of the batches to be created for the dataloaders.

    Returns:
        A tuple of dataloaders for the train and test datasets, a list
        containing the names of the images labels, and a bool indicating
        whether the function was executed as expected.

        If last value within the return is a False, should not use the other
        values.
    """
    # Create datasets
    logging.info("Creating Datasets.")

    if not image_path.is_dir():
        logging.error(f"There is no folder: '{image_path}'.")
        err_dataloader = DataLoader(ErrorDataset())
        return ((err_dataloader, err_dataloader), [], False)

    try:
        train_dataset = datasets.ImageFolder(
            root=image_path / "train",
            transform=train_transform)
        test_dataset = datasets.ImageFolder(
            root=image_path / "test",
            transform=test_transform)
        image_classes_list = train_dataset.classes

    except Exception as e:
        logging.error(f"Error loading images from: '{image_path}'.")
        err_dataloader = DataLoader(ErrorDataset())
        return ((err_dataloader, err_dataloader), [], False)

    # Turn datasets into dataloaders
    logging.info("Turning train and test datasets into dataloaders.")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True)

    return ((train_dataloader, test_dataloader), image_classes_list, True)
