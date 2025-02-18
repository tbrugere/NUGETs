from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import DistanceBatch
from nugets.models.model import EncoderDecoder

class DistanceEncoderDecoder(EncoderDecoder):
    """Identity encoder-decoder"""

    def __init__(self, input_dim: tuple[int, int], backbone_input_dim: tuple[int, int],
                 backbone_output_dim: int, same_input_proj=True):
        super().__init__()
        in1, in2 = input_dim
        back_in1, back_in2 = backbone_input_dim
        if same_input_proj: 
            assert in1 == in2
            assert back_in1 == back_in2
        self.in_proj1 = Linear(in1, back_in1)
        self.in_proj2 = Linear(in2, back_in2) if not same_input_proj else self.in_proj1
        self.out_proj = Linear(backbone_output_dim, 1)


    def encode(self, batch: DistanceBatch):
        return (self.in_proj1(batch.set1), self.in_proj2(batch.set2)), None

    def decode(self, result: Tensor):
        return self.out_proj(result).squeeze(-1)

    def compute_loss(self, batch: DistanceBatch, backbone_result: Tensor, encoder_info):
        loss = mse_loss(backbone_result, batch.distance)
        return loss

