
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNPrenet(torch.nn.Module):
    def __init__(self):
        super(CNNPrenet, self).__init__()

        # Define the layers using Sequential container
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
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



class CNNDecoderPrenet(nn.Module):
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
        x = self.ln1(x.permute(0, 2, 1)).permute(0, 2, 1)  # Transpose to [batch_size, feature_dim, sequence_length]
        x = self.tanh(x)
        x = self.dropout1(x)

        for i in range(len(self.convs_intermedite)):
            x = self.convs_intermedite[i](x)
        x = self.ln2(x.permute(0, 2, 1)).permute(0, 2, 1)  # Transpose to [batch_size, feature_dim, sequence_length]
        x = self.tanh(x)
        x = self.dropout2(x)

        x = self.conv_post(x)
        x = self.ln3(x.permute(0, 2, 1)).permute(0, 2, 1)  # Transpose to [batch_size, feature_dim, sequence_length]
        x = self.dropout3(x)

        return x


class ScaledPositionalEncoding(nn.Module):
    """
    This class implements the absolute sinusoidal positional encoding function
    with an adaptive weight parameter alpha.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd channels (got channels={input_size})"
            )
        self.max_len = max_len
        self.alpha = nn.Parameter(torch.ones(1))  # Define alpha as a trainable parameter
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        pe_scaled = self.pe[:, :x.size(1)].clone().detach() * self.alpha  # Scale positional encoding by alpha
        return pe_scaled

