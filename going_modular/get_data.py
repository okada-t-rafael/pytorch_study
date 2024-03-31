"""
Contains functions for downloading an image dataset and unzip it.
"""
import logging
import requests
import zipfile
from pathlib import Path
from typing import Tuple


def download_zip_dataset(
        dataset_url: str,
        folder_name: str,
        ) -> Tuple[Path, bool]:
    """Downloads an image dataset from a given url and unzip it.

    The files within the zip must respect the following structure:
    .
    |-- root_folder
        |-- test
        |   |-- class_one
        |   |-- class_two
        `-- train
            |-- class_one
            |-- class_two

    Args:
        dataset_url: An URL of a zip file containing images for each class
            divided into train and test subfolders.
        folder_name: The name of the folder to be creatd when unzipping the
            zip files.

    Returns:
        The path where the image were download and a boolean indicating whether
        everything was executed as expected. For example:

        (PosixPath("folder_name"), True)

        If last value within the return is a False, should not use the other
        values.
    """
    # Setup path to a data folder
    data_path: Path = Path("data")
    image_path: Path = data_path / folder_name

    # Check whether the image folder already exists, if now download it...
    if image_path.is_dir():
        logging.info(
            f"Directory '{image_path}' already exists. Skipping download.")
        return image_path, True

    logging.info(f"Creating directory: '{image_path}'")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download zipfile
    zipfile_name = dataset_url.split("/")[-1]
    try:
        logging.info(f"Downloading: '{zipfile_name}'")
        req = requests.get(dataset_url)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error while downloading '{dataset_url}': {e}")
        return Path(""), False

    # Saving downloaded file
    try:
        with open(data_path / zipfile_name, "wb") as f:
            f.write(req.content)
    except IOError as e:
        logging.error(f"Error while writing '{zipfile_name}': {e}")
        return Path(""), False

    # Unzip images
    try:
        with zipfile.ZipFile(data_path / zipfile_name, "r") as zip_ref:
            logging.info(f"Unzipping: '{zipfile_name}'")
            zip_ref.extractall(image_path)
    except Exception as e:
        logging.error(f"Error while unzipping '{zipfile_name}': {e}")
        return Path(""), False

    return image_path, True
