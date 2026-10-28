from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Transform, Datapoint
# import numpy as np
import ot
import torch
from torch import Tensor
from torch_heterogeneous_batching import Batch
from nugets.datasets.datapoint_types import Set_batch, Set_datapoint
from nugets.models.backbone import BackBone
from nugets.losses.losses import directional_width_loss
import warnings
from torch_geometric.utils import softmax


from .task import Task
from .register import register
from .transforms import SetLabelTransform
# from .losses import directional_width_loss

@register
class EpsilonKernelTask(Task):
    def __init__(self, *args, kernel_size=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size 

    def process_dataset(self, dataset):
        return dataset
    
    def datapoint_type(self):
        return Set_datapoint

    def compute_metrics(self, datapoint: Set_datapoint, results: Batch):
        target = datapoint.pointset
        coreset = Batch.from_list([
            points[logits.argmax(dim=0)]
            for points, logits in zip(target.batch_split(), results.batch_split())
        ], order=1)
        return dict(hard_directional_width_error=directional_width_loss(
            predicted=coreset, target=target, in_dim=target.data.shape[-1]))

    def get_encoder_decoder(self, backbone:BackBone, loss_function: str="directional_width_loss", **kwargs):

        from nugets.models.encoder_decoders.epsilon_kernel import EpsilonKernelEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        if loss_function != "directional_width_loss":
            warnings.warn("Only directional_width_loss compatible with this task.")
        return EpsilonKernelEncoderDecoder(input_dim=dataset_info["dim"],
                                                   backbone_input_dim=backbone_input_dim,
                                                   backbone_output_dim=backbone_output_dim,
                                                   output_dim = self.kernel_size,
                                                   loss_function="directional_width_loss")
