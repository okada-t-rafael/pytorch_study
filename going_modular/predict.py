import argparse
import logging
import model_builder
import torch
import torchvision

# Creating a parser
parser = argparse.ArgumentParser()

# Get an image path
parser.add_argument(
    "--image",
    help="target image filepath to predict on")

# Get a model path
parser.add_argument(
    "--model_path",
    default="models/tinyvgg_model.pth",
    type=str,
    help="target model to use for prediction")

args = parser.parse_args()

# Setup class names
class_name_list = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the image path
IMAGE = args.image
logging.info(f"Predicting on {IMAGE}")

# Load in the model
model = model_builder.TinyVGG(
    in_channels=3,
    hidden_units=20,
    out_features=len(class_name_list),
    img_shape=(64, 64))

model.load_state_dict(torch.load(args.model_path))

# Data preparation
image = torchvision.io.read_image(str(IMAGE)).type(torch.float32)
image = image / 255.0

transform = torchvision.transforms.Resize(size=(64, 64))
image = transform(image)

# Predict on image
model.eval()
with torch.inference_mode():
    # Send image to target device
    image = image.to(device)

    # Get pred logits
    pred_logits = model(image.unsqueeze(dim=0))
    pred_prob = torch.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)

print(f"Pred class: {class_name_list[pred_label]}, Pred prob: {pred_prob.max():.3f}")  # noqa: E501
