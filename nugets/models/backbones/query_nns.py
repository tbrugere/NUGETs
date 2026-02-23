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

from torch_geometric.nn.resolver import aggregation_resolver

@register
class QueryNetwork(BackBone):
    """
    Backbone for query architecture.

    The encoder for the object being queried (i.e. the point cloud or the range) can be any problem.

    The main difference here is that this backbone takes a tuple as the argument for the forward pass. 
    This will wrap another architecture that operates on the object. 
    """
    encoder: InnerBackbone = other_backbone_hyperparameter("backbone for the encoder")

    set_encoder: BackBone = model_attribute()

    def __setup__(self):
        self.set_encoder = self.encoder.load()

    def forward(self, query_tuple: tuple[Batch, torch.Tensor], return_reg_loss: bool = False):
        set_vec, _ = self.set_encoder(query_tuple[0])
        return (set_vec, query_tuple[1]), None
    
    def get_input_dim(self):
        return self.set_encoder.get_input_dim()
    def get_output_dim(self):
        return self.set_encoder.get_output_dim()
