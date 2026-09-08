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


from .task import Task
from .register import register
from .transforms import SetLabelTransform
# from .losses import directional_width_loss

@register
class EpsilonKernelTask(Task):
    def process_dataset(self, dataset):
        return dataset
    
    def datapoint_type(self):
        return Set_datapoint

    def compute_metrics(self, datapoint: Set_datapoint, results: Batch):
        target = datapoint.pointset
        dim = self.dataset_info()["dim"]
        directional_width = directional_width_loss(target, results, in_dim=dim)
        return dict(directional_width_error = directional_width)

    def get_encoder_decoder(self, backbone:BackBone, loss_function: str="directional_width_loss", **kwargs):

        from nugets.models.encoder_decoders.epsilon_kernel import EpsilonKernelIdentityEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = dataset_info["dim"]
        backbone_output_dim = dataset_info["dim"]
        if loss_function != "directional_width_loss":
            warnings.warn("Only directional_width_loss compatible with this task.")
        return EpsilonKernelIdentityEncoderDecoder(input_dim=dataset_info["dim"],
                                                   backbone_input_dim=backbone_input_dim,
                                                   backbone_output_dim=backbone_output_dim,
                                                   output_dim = dataset_info["dim"],
                                                   loss_function="directional_width_loss")