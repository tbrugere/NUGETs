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

        centers = backbone_result[:, :input_dim] 
        radius = backbone_result[:, input_dim]

        predicted_center  = batch.label[:, :input_dim]
        predicted_radius = batch.label[:, input_dim]
        return torch.sum(torch.linalg.norm(predicted_center - centers, axis=1)) + mse_loss(radius, predicted_radius)

class MECEncoderDecoder(EncoderDecoderWithProjection):
    """ Minimum Enclosing Cylinder """

    def compute_loss(self, result: Tensor):
        return 0

class MEAEncoderDecoder(EncoderDecoderWithProjection):
    """ Minimum Enclosing Annulus """

    # output d-dimensional center and two radii
    def compute_loss(self, result: Tensor):
        return 