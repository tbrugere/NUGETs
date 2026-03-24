from typing import TypeAlias
from torch import Tensor, randn_like
from torch import sum as tensor_sum
from torch import cat as concatenate
from torch.nn import Linear, Sequential, LeakyReLU
from torch.linalg import norm
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
                loss_function:str, absolute_positional_encoding: str | None = None, inject_noise: int = 0, *args, **kwargs):
        assert backbone_output_dim == backbone_input_dim 
        super().__init__()
        self.in_proj = Linear(input_dim, backbone_input_dim)
        self.out_proj = Linear(backbone_output_dim, output_dim)
        self.loss_function = getattr(Losses, loss_function)
        self.alpha = inject_noise

    def encode(self, batch:QueryBatch):
        # TODO: Add positional encoding support here. 
        ptset = batch.pointset
        labelset = batch.label
        queryset = batch.queryset
        # Transformed query vectors returned as encoder_info
        return (self.in_proj(batch.pointset), self.in_proj(queryset)), None

class ApproximateQueryEncoderDecoder(EncoderDecoder):
    """
    This encoder/decoder architecture is used for directly outputting 
    approximations of the query. 
    """

    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int|None,
                loss_function:str, absolute_positional_encoding: str | None = None, inject_noise: int = 0, *args, **kwargs):
        assert backbone_output_dim == backbone_input_dim 
        super().__init__()
        self.in_proj = Linear(input_dim, backbone_input_dim)
        # self.out_proj = Linear(backbone_output_dim, output_dim)
        self.out_proj = Sequential(Linear(backbone_output_dim*2, backbone_output_dim * 2), 
                                   LeakyReLU(), 
                                   Linear(backbone_output_dim*2, backbone_output_dim * 2),
                                   LeakyReLU(),
                                   Linear(backbone_output_dim*2, output_dim))
        self.loss_function = getattr(Losses, loss_function)
        self.alpha = inject_noise
    
    def encode(self, batch:QueryBatch):
        # TODO: Add positional encoding support here. 
        ptset = batch.pointset
        labelset = batch.label
        queryset = batch.queryset
        # Transformed query vectors returned as encoder_info
        return (self.in_proj(batch.pointset), self.in_proj(queryset)), None


    def decode(self, result: QueryBackboneResult):
        """
        
        """
        global_representation = result[0].mean()
        concatenated_rep = concatenate((global_representation, result[1]), dim=1)
        logits = self.out_proj(concatenated_rep)
        return logits.squeeze()
    
    def compute_loss(self, batch:QueryBatch, backbone_result: QueryBackboneResult, encoder_info):  
        logits = self.decode(backbone_result)
        return self.loss_function(input=logits, target=batch.label.float(), reduction="mean") # TODO: make sure only binary cross entropy works here. 


class SetToPointRegressionEncoderDecoder(QueryEncoderDecoder):
    def decode(self, result: QueryBackboneResult):
        """
        Return logits.
        TODO: Add in more complex decoding process here. 
        1. q \in \R^d, [x_1, \dots ,x_n] \in \R^(n x d)
        2. change from projection to something else? 

        1. Score points with normalized cosine similarity or normalized negative squared distance? 
        1a. Make sure that logits do not explode. Negative squared distance may be smoother than just a 
        raw dot product. 
        1b. Yusu suggests: change to e^{-x}. Similarity vs. closeness, the network may have a hard time switching
        from a similarity metric (such as dot product) to a closeness parameter. 
        2. Add in a fixed temperature parameter to control softmax sharpness. 
        3. The output from the backbone may also need to be normalized in general. This may help stabilize 
        the training. At least normalize the query embedding. 
        """

        result_set = self.out_proj(result[0])
        result_query = self.out_proj(result[1])
        result_norm = norm(result_query, ord=2, dim=0)
        normalized_query = result_query/result_norm

        idx = result_set.batch
        res = result_set * normalized_query[idx]
        logits = res.data.sum(axis=1)
        return logits

    def compute_loss(self, batch: QueryBatch, backbone_result: QueryBackboneResult, encoder_info):
        """
        Compute loss by taking the softmax of the logits and then return the nearest neighbor predictions. 
        """
        logits = self.decode(backbone_result)
        if self.alpha > 0 and self.training:
            logits = self.alpha * randn_like(logits) + logits
        probs = softmax(src=logits, index=batch.pointset.batch)
        scaled_pts = probs.unsqueeze(1)*batch.pointset.data
        nn_pred = scatter(src=scaled_pts, index=batch.pointset.batch, reduce='sum', dim=0)
        return self.loss_function(input=nn_pred, target=batch.label)
        
