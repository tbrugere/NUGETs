from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import Set_batch,SetToLabelSetBatch
from nugets.models.model import EncoderDecoderWithProjection
import nugets.losses.losses as Losses
from torch_geometric.utils import softmax


class SetMembershipEncoderDecoder(EncoderDecoderWithProjection):
    def __init__(self, loss_function: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_losses = ["scatter_binary_cross_entropy", "scatter_binary_focal_loss"]
        if loss_function not in self.allowed_losses:
            raise ValueError(f"only losses in {self.allowed_losses} allowed for set membership")
        self.loss_function = getattr(Losses, loss_function)

    def decode(self, backbone_result: Batch):
        out=self.out_proj(backbone_result)
        # result=softmax(src=out.data, index=out.batch)
        return out

    def encode(self, batch: SetToLabelSetBatch):
        return self.in_proj(batch.pointset), None

    def compute_loss(self, batch: SetToLabelSetBatch, backbone_result: Batch, encoder_info) -> Tensor:
        decoder_result = self.decode(backbone_result)
        batch_index = batch.labelset.batch
        membership_encoding = batch.labelset.data
        return self.loss_function(predicted=decoder_result.data.squeeze(1), target=membership_encoding.squeeze(1), index=batch_index )
