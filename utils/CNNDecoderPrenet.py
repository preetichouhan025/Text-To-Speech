
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNDecoderPrenet(nn.Module):
    """
    A CNN-based pre-net decoder module.

    This module takes input tensors and processes them through linear layers with ReLU activation and dropout
    to prepare the input for further processing in a decoder network.

    Args:
        input_dim (int): The dimensionality of the input tensor. Default is 80.
        hidden_dim (int): The dimensionality of the hidden layer in the linear transformations. Default is 256.
        output_dim (int): The dimensionality of the output tensor from the linear transformations. Default is 256.
        final_dim (int): The final dimensionality of the output tensor after linear projection. Default is 512.
        dropout_rate (float): The dropout rate to be applied. Default is 0.5.

    Inputs:
        x (torch.Tensor): A tensor of shape (batch_size, input_dim, sequence_length) representing the input features.

    Outputs:
        torch.Tensor: A tensor of shape (batch_size, final_dim, sequence_length) representing the processed features.

    Examples:
        # Initialize the decoder
        >>> decoder = CNNDecoderPrenet()
        # Create input tensor
        >>> input_tensor = torch.randn(4, 80, 10)
        # Forward pass
        >>> output = decoder(input_tensor)
        # Output tensor shape: torch.Size([4, 512, 10])
    """

    def __init__(self, input_dim=80, hidden_dim=256, output_dim=256, final_dim=512, dropout_rate=0.5):
        super(CNNDecoderPrenet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.linear_projection = nn.Linear(output_dim, final_dim) # Added linear projection
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

      # Transpose the input tensor to have the feature dimension as the last dimension
      x = x.transpose(1, 2)
      # Apply the linear layers
      x = F.relu(self.layer1(x))
      x = self.dropout(x)
      x = F.relu(self.layer2(x))
      x = self.dropout(x)
      # Apply the linear projection
      x = self.linear_projection(x)
      x = x.transpose(1, 2)

      return x
