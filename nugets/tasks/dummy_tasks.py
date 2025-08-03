from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Datapoint
from ml_lib.datasets.transforms import FunctionTransform
from ml_lib.datasets import Transform

from torch import Tensor, tensor
from torch_heterogeneous_batching import Batch

from .task import Task
from .register import register
from nugets.models.backbone import BackBone
from nugets.models.encoder_decoders.identity import SetIdentityEncoderDecoder, SingleLabelSetEncoderDecoder, SingleLabelGraphEncoderDecoder
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint, LabeledGraphDatapoint


@register
class SetIdentityTask(Task):
    def process_dataset(self, dataset):
        return dataset

    def compute_metrics(self, datapoint, results):
        del datapoint, results
        return dict(error=tensor(1.0))

    def get_encoder_decoder(self, backbone: BackBone):
        input_dim = self.dataset_info()["dim"]
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        return SetIdentityEncoderDecoder(input_dim=input_dim, 
                                         backbone_input_dim=backbone_input_dim,
                                         backbone_output_dim=backbone_output_dim,)

class SingleLabel(Transform):
    label: Callable[Tensor, Tensor]

    def __init__(self):
        pass

    def __len__(self):
        return len(self.inner)
    
    def __getitem__(self, idx):
        pointset = self.inner[idx].pointset
        label = tensor([1.0, 0.0])
        return LabeledSetDatapoint(pointset=pointset, label=label)

@register
class SingleLabelDummyTask(Task):
    def process_dataset(self, dataset):
        transform = SingleLabel()
        return transform(dataset)
    
    def compute_metrics(self, datapoint, results):
        del datapoint, results
        return dict(error=tensor(1.0))
    
    def datapoint_type(self):
        return LabeledSetDatapoint
    
    def get_encoder_decoder(self, backbone: BackBone):

        input_dim = self.dataset_info()["dim"]
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        output_dim = 2
        return SingleLabelSetEncoderDecoder(input_dim=input_dim, 
                                            backbone_input_dim=backbone_input_dim, 
                                            backbone_output_dim=backbone_output_dim, 
                                            output_dim=output_dim)

class SingleGraphLabel(Transform):
    def __getitem__(self, idx):
        graph = self.inner[idx].graph
        label = tensor(0)
        return LabeledGraphDatapoint(graph=graph, label=label)

    def __len__(self):
        return len(self.inner)


@register

class SingleLabelGraphDummyTask(Task):
    def process_dataset(self, dataset):
        return SingleGraphLabel()(dataset)

    def datapoint_type(self):
        return LabeledGraphDatapoint

    def compute_metrics(self, datapoint, results):
        del datapoint, results
        return dict(error=tensor(1.0))

    def get_encoder_decoder(self, backbone: BackBone):
        input_dim = self.dataset_info()["dim"]
        backbone_input_dim = backbone.get_inut_dim()
        backbone_output_dim = backbone.get_output_dim()
        output_dim = 2
        return SingleLabelGraphEncoderDecoder(input_dim=input_dim, 
                                            backbone_input_dim=backbone_input_dim, 
                                            backbone_output_dim=backbone_output_dim, 
                                            output_dim=output_dim)
    

    
