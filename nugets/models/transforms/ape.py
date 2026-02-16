import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor
from ml_lib.datasets.datapoint import Datapoint

from .register import register as transform_register
from .transform import PositionalEncodingTransform



@transform_register
class SinusoidalAbsolutePositionalEncoding(PositionalEncodingTransform):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Datapoint):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        ptcloud = x.pointset
        x = ptcloud + self.pe[:ptcloud.size(0)]
        return x