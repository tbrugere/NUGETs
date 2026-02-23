from typing import TypeAlias
from torch import Tensor
from torch import sum as tensor_sum
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import Set_batch
from nugets.models.model import EncoderDecoder
import nugets.losses.losses as Losses
from nugets.datasets.datapoint_types import QueryDatapoint, QueryBatch

import sys 

QueryBackboneResult: TypeAlias = Tensor | tuple[Batch|Tensor, Tensor]

class QueryEncoderDecoder(EncoderDecoder):

    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int|None,
                loss_function:str, absolute_positional_encoding: str | None = None, *args, **kwargs):
        assert backbone_output_dim == backbone_input_dim 
        super().__init__()
        self.in_proj = Linear(input_dim, backbone_input_dim)
        self.out_proj = Linear(backbone_output_dim, output_dim)

    def encode(self, batch:QueryBatch):
        # TODO: Add positional encoding support here. 
        ptset = batch.pointset
        labelset = batch.label
        queryset = batch.queryset
        # Transformed query vectors returned as encoder_info
        return (self.in_proj(batch.pointset), self.in_proj(queryset)), None
    

class SetQueryEncoderDecoder(QueryEncoderDecoder):
    def decode(self, result):
        
        backbone_result_set = result[0]
        backbone_result_query = result[1]
        idx = backbone_result_set.batch
        res = backbone_result_set*backbone_result_query[idx]
        val = res.data.sum(axis=1).unsqueeze(1)
        res.data = val
        return val
        
    def compute_loss(self, batch:QueryBatch, backbone_result:QueryBackboneResult, encoder_info):
        return 0.0

class RangeQueryEncoderDecoder(QueryEncoderDecoder):
    def decode(self, result: QueryBackboneResult):
        backbone_result_set = result[0]
        backbone_result_query = result[1]
        return backbone_result_query.transpose() @ backbone_result_set.mean()

    def compute_loss(self, batch:QueryBatch, backbone_result: QueryBackboneResult, encoder_info):
        return 0.0