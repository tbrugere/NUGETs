from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Transform, Datapoint
import numpy as np
import ot
import torch
from torch import Tensor
from torch_heterogeneous_batching import Batch
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint, SetToLabelSetBatch, SetToLabelSetDatapoint
from nugets.models.backbone import BackBone
from nugets.losses.losses import scatter_binary_cross_entropy
from scipy.spatial import ConvexHull

from .task import Task
from .register import register
from .transforms import SetLabelTransform, SetToLabelSetTransform



@register
class ConvexHullMembershipTask(Task):
    """
    Predict membership in the convex hull. Returns a one-hot encoding corresponding to points in the convex hull. 
    """
    def process_dataset(self, dataset):
        transform = SetToLabelSetTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return SetToLabelSetDatapoint
    
    def label(self, pointset):
        raw_pointcloud = pointset.numpy()
        out = ConvexHull(raw_pointcloud)
        label = np.zeros(raw_pointcloud.shape[0])
        label[out.vertices] = 1.0
        out = torch.tensor(label).unsqueeze(1)
        return out.float()
    
    def compute_metrics(self, datapoint: SetToLabelSetDatapoint, results: Batch):
        # TODO: Add more logging metrics here
        cvx_hull = datapoint.labelset.data.squeeze(1)
        predicted_membership=results.data.squeeze(1)
        batch_index = datapoint.labelset.batch
        result = scatter_binary_cross_entropy(predicted=predicted_membership, target=cvx_hull, index=batch_index)
        return dict(BCE_loss=result)
    
    def get_encoder_decoder(self, backbone: BackBone, loss_function:str, **kwargs):
        from nugets.models.encoder_decoders.set_membership import SetMembershipEncoderDecoder
        dataset_info=self.dataset_info()
        backbone_input_dim=backbone.get_input_dim()
        backbone_output_dim=backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = 1 # Each point should output a single logit
        return SetMembershipEncoderDecoder(input_dim=input_dim, 
                                           backbone_input_dim=backbone_input_dim, 
                                           backbone_output_dim=backbone_output_dim,
                                           output_dim=output_dim,
                                           loss_function=loss_function)

@register
class ConvexHullTask(Task):
    def process_dataset(self, dataset):
        raise NotImplementedError
    def compute_metrics(self, datapoint, results):
        raise NotImplementedError
    def get_encoder_decoder(self, backbone:BackBone):
        raise NotImplementedError 