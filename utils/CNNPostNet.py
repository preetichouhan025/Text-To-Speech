
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNPostNet(torch.nn.Module):
    """
    Conv Postnet
    Arguments
    ---------
    n_mel_channels: int
       input feature dimension for convolution layers
    postnet_embedding_dim: int
       output feature dimension for convolution layers
    postnet_kernel_size: int
       postnet convolution kernal size
    postnet_n_convolutions: int
       number of convolution layers
    postnet_dropout: float
        dropout probability fot postnet
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        postnet_dropout=0.1,
    ):
        super(CNNPostNet, self).__init__()

        self.conv_pre = nn.Conv1d(
            in_channels=n_mel_channels,
            out_channels=postnet_embedding_dim,
            kernel_size=postnet_kernel_size,
            padding="same",
        )

        self.convs_intermedite = nn.ModuleList()
        for i in range(1, postnet_n_convolutions - 1):
            self.convs_intermedite.append(
                nn.Conv1d(
                    in_channels=postnet_embedding_dim,
                    out_channels=postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding="same",
                ),
            )

        self.conv_post = nn.Conv1d(
            in_channels=postnet_embedding_dim,
            out_channels=n_mel_channels,
            kernel_size=postnet_kernel_size,
            padding="same",
        )

        self.tanh = nn.Tanh()
        self.ln1 = nn.LayerNorm(postnet_embedding_dim)
        self.ln2 = nn.LayerNorm(postnet_embedding_dim)
        self.ln3 = nn.LayerNorm(n_mel_channels)
        self.dropout1 = nn.Dropout(postnet_dropout)
        self.dropout2 = nn.Dropout(postnet_dropout)
        self.dropout3 = nn.Dropout(postnet_dropout)


    def forward(self, x):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            a (batch, time_steps, features) input tensor
        Returns
        -------
        output: torch.Tensor (the spectrogram predicted)
        """
        x = self.conv_pre(x)
        x = self.ln1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.tanh(x)
        x = self.dropout1(x)

        for i in range(len(self.convs_intermedite)):
            x = self.convs_intermedite[i](x)
        x = self.ln2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.tanh(x)
        x = self.dropout2(x)

        x = self.conv_post(x)
        x = self.ln3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout3(x)

        return x
