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

from torch_geometric.utils import softmax
from torch_scatter import scatter

import sys 

QueryBackboneResult: TypeAlias = Tensor | tuple[Batch|Tensor, Tensor]

class QueryEncoderDecoder(EncoderDecoder):

    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int|None,
                loss_function:str, absolute_positional_encoding: str | None = None, *args, **kwargs):
        assert backbone_output_dim == backbone_input_dim 
        super().__init__()
        self.in_proj = Linear(input_dim, backbone_input_dim)
        self.out_proj = Linear(backbone_output_dim, output_dim)
        self.loss_function = getattr(Losses, loss_function)

    def encode(self, batch:QueryBatch):
        # TODO: Add positional encoding support here. 
        ptset = batch.pointset
        labelset = batch.label
        queryset = batch.queryset
        # Transformed query vectors returned as encoder_info
        return (self.in_proj(batch.pointset), self.in_proj(queryset)), None

class RangeQueryEncoderDecoder(QueryEncoderDecoder):
    def decode(self, result: QueryBackboneResult):
        """
        Decode range query task by taking the doct product between the query vector and the 
        global representation of the range. 

        Output: logits which represent whether each query is in the polygon or not
        """
        result_set = self.out_proj(result[0])
        result_query = self.out_proj(result[1])
        global_representation = result_set.mean()
        logits = (result_query * global_representation).sum(dim=1)
        return logits
    
    def compute_loss(self, batch:QueryBatch, backbone_result: QueryBackboneResult, encoder_info):  
        logits = self.decode(backbone_result)
        return self.loss_function(input=logits, target=batch.label.float(), reduction="mean") # TODO: make sure only binary cross entropy works here. 
    

class SetToPointRegressionEncoderDecoder(QueryEncoderDecoder):
    def decode(self, result: QueryBackboneResult):
        """
        Return logits.
        TODO: Add in more complex decoding process here. 
        """
        result_set = self.out_proj(result[0])
        result_query = self.out_proj(result[1])
        idx = result_set.batch
        res = result_set * result_query[idx]
        logits = res.data.sum(axis=1)
        return logits

    def compute_loss(self, batch: QueryBatch, backbone_result: QueryBackboneResult, encoder_info):
        """
        Compute loss by taking the softmax of the logits and then return the nearest neighbor predictions. 
        """
        logits = self.decode(backbone_result)
        probs = softmax(src=logits, index=batch.pointset.batch)
        scaled_pts = probs.unsqueeze(1)*batch.pointset.data
        nn_pred = scatter(src=scaled_pts, index=batch.pointset.batch, reduce='sum', dim=0)
        return self.loss_function(input=nn_pred, target=batch.label)
        
### The following two classes should not be used. Eventually, I will remove them altogether. 
class SetMembershipQueryEncoderDecoder(QueryEncoderDecoder):

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
