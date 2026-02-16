import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor

class PositionalEncodingTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()