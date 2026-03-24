from typing import Callable
from ml_lib.models.layers import MLP
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.batch import BatchIndicator
import torch
from torch import nn

from nugets.models.backbone import (BackBone, int_hyperparameter, bool_hyperparameter, 
                model_attribute, hyperparameter,  other_backbone_hyperparameter, InnerBackbone)
from nugets.models.backbones.register import register
import nugets.losses.losses as Losses

from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.utils import softmax


@register
class EpsilonKernelNetwork(BackBone):
    """
    Epsilon Kernel network backbone for point clouds

    Like the query model, this network should take other models as a backbone. 

    Output: 
        Let d_in and d_out be the input and output dimensions of the model. This model outputs a 
        tensor of shape (Bd_out, d_in) where B is the batch size.
        For each input point cloud, the model should output a point cloud of size d_out. 
    """

    encoder: InnerBackbone = other_backbone_hyperparameter("backbone for the encoder")

    set_encoder: BackBone = model_attribute()

    def __setup__(self):
        self.set_encoder: BackBone = model_attribute()
    

    def forward(self, batch: Batch, return_reg_loss: bool=False):
        del return_reg_loss

        out = self.set_encoder(batch)
        out = softmax(src=out, index=out.batch)

        # TODO: There should be a smarter/faster way to implement this approximation. Maybe with torch.einsum?
        coresets = []
        start = 0
        for num in batch.n_nodes:
            end = start + num
            ptset = batch.data[start:end]
            ptset_probs = out.data[start:end] 
            coreset = torch.mm(ptset_probs.T, ptset)
            coresets.append(coreset)
            start = end

        output_ptset = Batch.from_list(coresets, order=1)
        return output_ptset, None