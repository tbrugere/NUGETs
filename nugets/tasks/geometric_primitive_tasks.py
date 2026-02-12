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
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint, SetToLabelSetBatch, SetToLabelSetDatapoint
from nugets.models.backbone import BackBone

from .task import Task
from .register import register
from .transforms import SetToLabelSetTransform


class ExtremalPointTask(Task):
    """
    This takes a set of points and a direction and outputs a one-hot encoding which indicates 
    whether it is an extremal point in that direction. 
    
    To represent the pointset and the direction, we will collate it all 
    together as a single point cloud. One additional dimension will be appended to each
    point to represent if it is part of the point cloud (0) or if it is a direction (1). 
    Additionally, each direction is randomly generated.

    """
    seed: int = 42 # Seed for randomly generated directions
    mean = np.ones(dim)
    covariance = np.identity(dim)

    def process_dataset(self, dataset):
        transform = SetToLabelSetTransform(self.label)
        return transform(dataset)
    
    def datapoint_type(self):
        return SetToLabelSetDatapoint
    
    def label(self, pointset):
        import numpy as np 
        rng = np.random.default_rng(self.seed)
        direction = rng.multivariate_normal(mean=self.mean, cov=self.covariance)
        direction = direction/np.linalg.norm(direction, p=2)
        idx = np.argmax(pointset @ direction)
        np.hstack((pointset, direction))
        raise NotImplementedError
    
    def get_encoder_decoder(self, backbone: BackBone, loss_function: str, **kwargs):
        from nugets.models.encoder_decoders.set_membership import SetMembershipEncoderDecoder
        dataset_info = self.dataset_info()
        input_dim = dataset_info["dim"]
        model_input_dim = input_dim + 1
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        output_dim = 1
        return SetMembershipEncoderDecoder(input_dim=input_dim, 
                                           backbone_input_dim=backbone_input_dim, 
                                           backbone_output_dim=backbone_output_dim,
                                           output_dim=output_dim,
                                           loss_function=loss_function)