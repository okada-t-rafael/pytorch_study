"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn
from typing import Tuple


class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN exaplainer website in
    Pytorch. See more: https://poloclub.github.io/cnn-explainer/
    """
    def __init__(
            self,
            in_channels: int,
            hidden_units: int,
            out_features: int,
            img_shape: Tuple[int, int]
            ) -> None:
        super().__init__()

        # First block
        self.conv_block_1, adj_img_shape = TinyVGG._create_conv_block(
            in_channels=in_channels,
            out_channels=hidden_units,
            img_shape=img_shape)

        # Second block
        self.conv_block_2, adj_img_shape = TinyVGG._create_conv_block(
            in_channels=hidden_units,
            out_channels=hidden_units,
            img_shape=adj_img_shape)

        # Classifier
        self.classifier = TinyVGG._create_classifier_block(
            in_features=hidden_units * adj_img_shape[0] * adj_img_shape[1],
            out_features=out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


    def _create_conv_block(
            in_channels: int,
            out_channels: int,
            img_shape: Tuple[int, int]
            ) -> Tuple[nn.Sequential, Tuple[int, int]]:
        """Creates a block of the neural network.

        This block includes two Conv2d and one MaxPool2d. And it also
        calculates the adjusted size of the 'image' for the flatten layer.

        Args:
            in_channels: The input size channel of the first conv layer.
            out_channels: The output size channel of the second conv layer.
            img_shape: The shape of the 'image' after passing through this
                block. It is influenced by the Conv2d layers and the MaxPool2d
                layer (see documentatin to adjust accordinly).

        Return:
            The instance of the block and the adjusted size of the 'image'.
        """
        # Create the conv block
        conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2))

        # Calculate the ajusted image shape
        img_x, img_y = img_shape
        img_x -= 2  # first Conv2d
        img_x -= 2  # second Conv2d
        img_x = int(img_x / 2)  # MaPool2d
        img_y -= 2  # first Conv2d
        img_y -= 2  # second Conv2d
        img_y = int(img_y / 2)  # MaPool2d

        return conv_block, (img_x, img_y)


    def _create_classifier_block(
            in_features: int,
            out_features: int,
            ) -> nn.Sequential:
        """Creates a classifier block to the neural network.

        It is composed by a flatten layer and a linear layer.

        Args:
            in_features: The number of features to be inserted in the linear
                layer. Note that this number must be ajusted to due to the
                flatten layer.
            out_feataures: Number of labels for the neural network to predict.

        Return:
            An instance of the block.
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
        )
