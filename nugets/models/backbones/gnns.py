from typing import Callable
import torch
from torch import nn
from torch_geometric.nn import GAT as GAT_nn

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
    n_layers: int = int_hyperparameter(description="number of layers")
    input_dim: int = int_hyperparameter(description="input dimension")
    output_dim: int = int_hyperparameter(description = "output dimension")

    feed_forward_hidden_dim: int=int_hyperparameter(description="number of hidden dimensions")

    def __setup__(self):
        self.gat = GAT_nn(in_channels = input_dim, 
                          out_channels = output_dim, 
                          num_layers=n_layers, 
                          heads=n_heads, 
                          hidden_channels=feed_forward_hidden_dim)
    
    def forward(self, x, edge_index):
        raise NotImplementedError("TODO: implement forward pass")

    def get_input_dim(self): return self.input_dim
    def get_output_dim(self): return self.output_dim

