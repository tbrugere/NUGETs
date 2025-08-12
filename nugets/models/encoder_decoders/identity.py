from torch import Tensor
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss

from nugets.datasets.datapoint_types import Set_batch, Point_datapoint, LabeledSetBatch, Graph_datapoint, LabeledGraphDatapoint
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
        decoded = self.decode(backbone_result)
        loss = self.loss(batch.pointset, decoded)
        return loss
    
class SingleLabelSetEncoderDecoder(EncoderDecoderWithProjection):
    """ Single label prediction encoder-decoder with cross entropy loss """
    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int):
        super().__init__(input_dim, backbone_input_dim, backbone_output_dim, output_dim)
        self.loss = CrossEntropyLoss()
    
    def encode(self, batch: LabeledSetBatch):
        return self.in_proj(batch.pointset), None
    
    def compute_loss(self, batch: LabeledSetBatch, backbone_result: Batch, encoder_info):
        del encoder_info
        decoded = self.decode(backbone_result)
        loss = self.loss(batch.label, decoded)
        return loss

class SingleLabelGraphEncoderDecoder(EncoderDecoderWithProjection):
    """ Single-label graph classification encoder-decoder"""

    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int):
        super().__init__(input_dim, backbone_input_dim, backbone_output_dim, output_dim)
        self.loss = CrossEntropyLoss()

    def encode(self, batch: LabeledGraphDatapoint):
        return self.in_proj(batch.pointset), None

    def compute_loss(self, batch: LabeledGraphDatapoint, backbone_result: Tensor, encoder_info):
        del encoder_info
        graph_embedding = backbone_result.mean(dim=0, keepdim=True)
        logits = self.decode(graph_embedding)
        label = batch.label.unsqueeze(0).long()
        loss = self.loss_fn(logits, label)
        return loss