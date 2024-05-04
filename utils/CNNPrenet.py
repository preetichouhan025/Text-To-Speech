
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNPrenet(torch.nn.Module):
    """A CNN-based pre-net module for feature extraction.

    This module consists of multiple convolutional layers followed by batch normalization,
    ReLU activation, and dropout for feature extraction.

    Args:
        None

    Attributes:
        conv_layers (nn.Sequential): Sequential container for convolutional layers.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

    Outputs:
        torch.Tensor: Output tensor of shape (batch_size, sequence_length, feature_dim).

    Example:
        >>> prenet = CNNPrenet()
        >>> input_tensor = torch.randn(32, 100)  # Input tensor of batch size 32 and sequence length 100
        >>> output_tensor = prenet(input_tensor)  # Extracted features with shape (32, 100, feature_dim)
    """

    def __init__(self):
        super(CNNPrenet, self).__init__()

        # Define the layers using Sequential container
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):

        # Add a new dimension for the channel
        x = x.unsqueeze(1)

        # Pass input through convolutional layers
        x = self.conv_layers(x)

        # Remove the channel dimension
        x = x.squeeze(1)

        # Scale the output to the range [-1, 1]
        x = torch.tanh(x)

        return x
