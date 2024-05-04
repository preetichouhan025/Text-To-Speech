
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    This code sample is taken from Speechbrain implementation
    with minor modifications
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
