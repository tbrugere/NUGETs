from torch import Tensor
from torch.nn.functional import mse_loss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import Set_batch, Point_datapoint
from nugets.models.model import EncoderDecoderWithProjection

class PointIdentityEncoderDecoder(EncoderDecoderWithProjection):
    """Identity encoder-decoder"""

    def encode(self, batch: Point_datapoint):
        return self.in_proj(batch.point), None

    def compute_loss(self, batch: Point_datapoint, backbone_result: Tensor, encoder_info):
        decoded = self.decode(backbone_result)
        loss = mse_loss(backbone_result, batch.point)
        return loss

class SetIdentityEncoderDecoder(EncoderDecoderWithProjection):
    """Identity encoder-decoder"""

    def __init__(self, input_dim: int, backbone_input_dim: int,
                 backbone_output_dim: int,):
        super().__init__(input_dim, backbone_input_dim, backbone_output_dim, input_dim)
        self.loss = BatchMSELoss()

    def encode(self, batch: Set_batch):
        return self.in_proj(batch.pointset), None

    def compute_loss(self, batch: Set_batch, backbone_result: Batch, encoder_info):
        del encoder_info
        loss = self.loss(batch.pointset, backbone_result)
        return loss
