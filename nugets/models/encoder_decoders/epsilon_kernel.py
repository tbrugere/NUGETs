from torch import Tensor
from torch import mm as matrix_multiply
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.nn.losses import BatchMSELoss
from torch_geometric.utils import softmax
import nugets.losses.losses as Losses


from nugets.datasets.datapoint_types import Set_batch, Point_datapoint, LabeledSetBatch
from nugets.models.model import EncoderDecoder
import sys


class EpsilonKernelEncoderDecoder(EncoderDecoder):
    def __init__(self, input_dim: int, backbone_input_dim: int, backbone_output_dim: int, output_dim: int,
                loss_function:str, absolute_positional_encoding: str | None = None, inject_noise: float = 0.1, *args, **kwargs):
        super().__init__()
        assert loss_function == "directional_width_loss" 
        self.loss_function = getattr(Losses, loss_function)
        self.input_dim = input_dim
        self.in_proj = Linear(input_dim, backbone_input_dim)
        self.out_proj = Linear(backbone_output_dim, output_dim)
    
    def encode(self, batch: Set_batch):
        return self.in_proj(batch.pointset), None
    
    def decode(self, result: Batch):
        out = self.out_proj(result)
        return out
    
    def compute_loss(self, batch: Set_batch, backbone_result: Batch, encoder_info):
        eps_kernel_logits = self.decode(backbone_result)
        eps_kernel_probs = softmax(src=eps_kernel_logits.data, index=eps_kernel_logits.batch)
        coresets = []
        start = 0
        for num in batch.pointset.n_nodes:
            end = start+num 
            ptset = batch.pointset.data[start:end]
            ptset_probs = eps_kernel_probs[start:end]
            coreset = matrix_multiply(ptset_probs.T, ptset)
            coresets.append(coreset)
            start = end
        eps_kernel = Batch.from_list(coresets, order=1)

        return self.loss_function(predicted=eps_kernel, target=batch.pointset, in_dim=self.input_dim)
