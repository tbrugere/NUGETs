import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor
from torch_heterogeneous_batching import Batch

from .register import register as transform_register
from .transform import PositionalEncodingTransform

@transform_register
class SinusoidalAbsolutePositionalEncoding(PositionalEncodingTransform):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, batch: Batch):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """ 
        # Per element in the batch, we need to add the positional encoding
        # self.pe needs to be indexed based on the elements in the batch. 
        idx = batch.batch
        idx_range = torch.arange(len(idx)).type_as(batch.ptr)
        pe_index = idx_range - batch.ptr[idx]
        return batch.data + self.pe[pe_index]
        