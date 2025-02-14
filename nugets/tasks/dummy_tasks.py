from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Datapoint
from ml_lib.datasets.transforms import FunctionTransform
from torch import Tensor
from torch_heterogeneous_batching import Batch

from .task import Task
from .register import register
from nugets.models.backbone import BackBone
from nugets.models.encoder_decoders.identity import SetIdentityEncoderDecoder


@register
class SetIdentityTask(Task):
    def process_dataset(self, dataset):
        return dataset

    def get_encoder_decoder(self, backbone: BackBone):
        input_dim = self.dataset_info()["dim"]
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        return SetIdentityEncoderDecoder(input_dim=input_dim, 
                                         backbone_input_dim=backbone_input_dim,
                                         backbone_output_dim=backbone_output_dim,)

    
