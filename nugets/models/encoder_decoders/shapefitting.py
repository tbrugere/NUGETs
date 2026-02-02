from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import Set_batch
from nugets.models.model import EncoderDecoderWithProjection, EncoderDecoderToVector
import nugets.losses.losses as Losses
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint



class MEBEncoderDecoder(EncoderDecoderToVector):

    def __init__(self, loss_function: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = getattr(Losses, loss_function)

    def encode(self, batch: LabeledSetBatch):
        return self.in_proj(batch.pointset), None
    
    def compute_loss(self, batch: LabeledSetBatch, backbone_result: Tensor, encoder_info):
        input_dim = batch.pointset.n_features
        result = self.decode(backbone_result)
        predicted_center = result[:, :input_dim] 
        predicted_radius = result[:, input_dim]

        center  = batch.label[:, :input_dim]
        radius = batch.label[:, input_dim]
        loss = self.loss_function(c=center, predicted_c=predicted_center, r=radius, predicted_r=predicted_radius )
        return loss

class MEAEncoderDecoder(EncoderDecoderToVector):
    """ Minimum Enclosing Annulus """

    def __init__(self, loss_function: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = getattr(Losses, loss_function)
    
    def encode(self, batch: LabeledSetBatch):
        return self.in_proj(batch.pointset), None

    # output d-dimensional center and two radii
    def compute_loss(self, 
                    batch: LabeledSetBatch,
                    backbone_result: Tensor,
                    encoder_info):
        input_dim = batch.pointset.n_features
        result = self.decode(backbone_result)
        predicted_centers = result[:, :input_dim]
        predicted_inner_radius = result[:, input_dim]
        predicted_outer_radius = result[:, input_dim + 1]

        centers = batch.label[:, :input_dim]
        inner_radius = batch.label[:, input_dim]
        outer_radius = batch.label[:, input_dim + 1]
        loss = self.loss_function(c=centers, predicted_c=predicted_centers, 
                                  inner_r=inner_radius, predicted_inner_r=predicted_inner_radius,
                                  outer_r=outer_radius, predicted_outer_r=predicted_outer_radius)
        return loss

class MEEEncoderDecoder(EncoderDecoderToVector):
    """ Minimum Enclosing Cylinder """
    def __init__(self, loss_function: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = getattr(Losses, loss_function)
    
    def encode(self, batch: LabeledSetBatch):
        return self.in_proj(batch.pointset), None
    def compute_loss(self, 
                    batch: LabeledSetBatch,
                    backbone_result: Tensor,
                    encoder_info):
        return 0
