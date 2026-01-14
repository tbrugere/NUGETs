from typing import TypeAlias
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import DistanceBatch
from nugets.models.model import EncoderDecoder
from nugets.losses.losses import SinkhornLoss

DistanceBackboneResult: TypeAlias = Tensor|tuple[Tensor, Batch|None, Batch|None]

class DistanceEncoderDecoder(EncoderDecoder):
    """Identity encoder-decoder"""

    def __init__(self, input_dim: tuple[int, int], backbone_input_dim: tuple[int, int],
                 backbone_output_dim: int, 
                 same_input_proj=True, 
                 backbone_reconstructs=False):
        super().__init__()
        in1, in2 = input_dim
        back_in1, back_in2 = backbone_input_dim
        if same_input_proj: 
            assert in1 == in2
            assert back_in1 == back_in2
        self.in_proj1 = Linear(in1, back_in1)
        self.in_proj2 = Linear(in2, back_in2) if not same_input_proj else self.in_proj1
        if backbone_reconstructs:
            self.sinkhorn_loss = SinkhornLoss()
            self.decode_proj1 = Linear(back_in1, in1)
            self.decode_proj2 = Linear(back_in2, in2) if not same_input_proj else self.decode_proj1
        else:
            self.sinkhorn_loss = self.decode_proj1 = self.decode_proj2 = None


    def encode(self, batch: DistanceBatch):
        return (self.in_proj1(batch.set1), self.in_proj2(batch.set2)), None

    def decode(self, result: DistanceBackboneResult):
        if isinstance(result, tuple):
            result, _, _ = result
        return result

    def compute_loss(self, batch: DistanceBatch, backbone_result: DistanceBackboneResult, encoder_info):
        if isinstance(backbone_result, tuple):
            assert self.sinkhorn_loss is not None
            assert self.decode_proj1 is not None and self.decode_proj2 is not None
            out_distances, out1, out2 = backbone_result
            assert out1 is not None and out2 is not None
            recon_loss = self.sinkhorn_loss(batch.set1, out1.map(self.decode_proj1)) \
                  + self.sinkhorn_loss(batch.set2, out2.map(self.decode_proj2))
        else:
            out_distances = backbone_result
            recon_loss = 0.
        loss = mse_loss(out_distances, batch.distance.float()) + recon_loss
        return loss

