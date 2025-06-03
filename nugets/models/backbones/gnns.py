from typing import Callable
from ml_lib.models.layers import MLP
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.batch import BatchIndicator
from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn
import torch
from torch import nn

from nugets.models.backbone import (BackBone, int_hyperparameter, bool_hyperparameter, 
                model_attribute, hyperparameter,  other_backbone_hyperparameter, InnerBackbone)
from nugets.models.backbones.register import register
import nugets.losses.losses as Losses

@register
class GAT(BackBone):
    """
    
    Graph attention network backbone

    """
    n_heads: int = int_hyperparameter(description="number of attention heads for GAT")
    n_layers: int = int_hyperparameters(description="number of layers")
    input_dim: int = int_hyperparameter(description="input dimension")
    output_dim: int = int_hyperparameter(description = "output dimension")

    feed_forward_hidden_dim: int=int_hyperparameter(description="number of hidden dimensions")

    # def __setup__(self):
    #     self.gnn = 

