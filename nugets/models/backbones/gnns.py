from typing import Callable
import torch
from torch import nn
from torch_geometric.nn import GATConv as GAT_nn # supports edge features

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
    feed_forward_hidden_dim: int = int_hyperparameter(description="number of hidden dimensions")
    edge_dim: int = int_hyperparameter(description="edge feature dimension")

    feed_forward_hidden_dim: int=int_hyperparameter(description="number of hidden dimensions")

    def __setup__(self):
        self.layers = nn.ModuleList()

        self.layers.append(GATConv(
            self.input_dim,
            self.feed_forward_hidden_dim,
            heads=self.n_heads,
            edge_dim=self.edge_dim,
            concat=True
            )
        )

        for _ in range(self.n_layers - 2):
            self.layers.append(
                GATConv(
                    self.feed_forward_hidden_dim * self.n_heads,
                    self.feed_forward_hidden_dim,
                    heads=self.n_heads,
                    edge_dim = self.edge_dim,
                    concat=True
                )
            )

        self.layers.append(GATConv(
            self.feed_forward_hidden_dim * self.n_heads,
            self.output_dim,
            heads=1,
            edge_dim=self.edge_dim,
            concat=False
            )
        )
    
    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)

        return x

    def get_input_dim(self): return self.input_dim
    def get_output_dim(self): return self.output_dim

