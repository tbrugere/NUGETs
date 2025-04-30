from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import Set_batch
from nugets.models.model import EncoderDecoderWithProjection



class MEBEncoderDecoder(EncoderDecoderWithProjection):
    """" Minimum Enclosing Ball """"
    
    # output d-dimensional center and a single value for the radius
    def compute_loss(self, 
                    batch: LabeledSetBatch, 
                    backbone_result: Tensor, 
                    encoder_info):
        input_dim = batch.n_features()

        predicted_center = backbone_result[:, :input_dim] 
        predicted_radius = backbone_result[:, input_dim]

        center  = batch.label[:, :input_dim]
        radius = batch.label[:, input_dim]
        return torch.sum(torch.linalg.norm(predicted_center - centers, axis=1)) + mse_loss(radius, predicted_radius)

class MECEncoderDecoder(EncoderDecoderWithProjection):
    """ Minimum Enclosing Cylinder """

    def compute_loss(self, 
                    batch: LabeledSetBatch,
                    backbone_result: Tensor,
                    encoder_info):
        return 0

class MEAEncoderDecoder(EncoderDecoderWithProjection):
    """ Minimum Enclosing Annulus """

    # output d-dimensional center and two radii
    def compute_loss(self, 
                    batch: LabeledSetBatch,
                    backbone_result: Tensor,
                    encoder_info):
        input_dim = batch.n_features()
        predicted_centers = backbone_result[:, :input_dim]
        predicted_inner_radius = backbone_result[:, input_dim]
        predicted_outer_radius = backbone_result[:, input_dim + 1]

        centers = batch.label[:, :input_dim]
        inner_radius = batch.label[:, input_dim]
        outer_radius = batch.label[:, input_dim + 1]
        return torch.sum(torch.linalg.norm(predicted_centers - centers, axis=1)) + mse_loss(inner_radius, predicted_inner_radius) + mse_loss(outer_radius, predicted_outer_radius)