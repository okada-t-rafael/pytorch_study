"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import data_setup
import engine
import get_data
import logging
import model_builder
import os
import torch
import utils
from pathlib import Path
from torchvision import transforms


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Add an arg for num_epochs
parser.add_argument(
    "--num_epochs",
    default=10,
    type=int,
    help="the number of epochs to train for")

# Add an arg for batch_size
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="numer of samples per batch")

# Add an arg for hidden_units
parser.add_argument(
    "--hidden_units",
    default=10,
    type=int,
    help="number of hidden units in hidden layers")

# Add an arg for learning_rate
parser.add_argument(
    "--learning_rate",
    default=0.001,
    type=float,
    help="learning rate to use for model")

# Add an arg for the image path
parser.add_argument(
    "--image_path",
    default="data/pizza_steak_sushi",
    type=str,
    help="image path in standard image classification format"
)

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
IMAGE_PATH = Path(args.image_path)

logging.info(
    f"Training a model epochs: {NUM_EPOCHS} \| "
    f"batch size: {BATCH_SIZE} \| hidden units: {HIDDEN_UNITS} \| "
    f"learning rate: {LEARNING_RATE}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
train_data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()])

test_data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64), antialias=True),
    transforms.ToTensor()])

# Create DataLoaders
dataloaders, class_name_list, ok = data_setup.create_dataloaders(
    image_path=IMAGE_PATH,
    train_transform=train_data_transform,
    test_transform=test_data_transform,
    batch_size=BATCH_SIZE)

# Create model
model = model_builder.TinyVGG(
    in_channels=3,
    hidden_units=HIDDEN_UNITS,
    out_features=len(class_name_list),
    img_shape=(64, 64))

# Set Loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE)

# Training
engine.train(
    model=model,
    train_dataloader=dataloaders[0],
    test_dataloader=dataloaders[1],
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device)

# Save
utils.save_model(
    model=model,
    target_dir="models",
    model_name="tinyvgg_model.pth")
