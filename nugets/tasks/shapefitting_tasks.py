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
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint
from nugets.models.backbone import BackBone

from .task import Task
from .register import register
from .transforms import SetLabelTransform
import miniball
import numpy as np


class ShapefittingTask(Task):
    """Task for fitting a shape to a set"""
    def process_dataset(self, dataset):
        transform = SetLabelTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return LabeledSetDatapoint
    
    def label(self, pointset):
        raise NotImplementedError

@register
class MinimumEnclosingBallTask(ShapefittingTask):
    """
    Task for fitting a ball to a point cloud 
    This works for any dimension input point cloud. 
    """
    def label(self, pointset):
        raw_pointcloud = pointset.numpy()
        C, r = miniball.get_bounding_ball(raw_pointcloud, epsilon=1e-6)
        return torch.tensor(np.append(C, r)).float()
    
    # TODO: Add more metrics for minimum enclosing ball 
    def compute_metrics(self, datapoint: LabeledSetDatapoint, results: Tensor):
        gt = datapoint.label
        gt_r = datapoint.label[:, -1]
        batch_radius_error = torch.abs(gt_r - results[:, -1])
        radius_relative_error = batch_radius_error/(gt_r + 1e-3)
        radius_mean_relative_error = radius_relative_error.mean()

        gt_center = datapoint.label[:, :-1]
        predicted_center = results[:, :-1]
        return dict(radius_relative_error=radius_mean_relative_error)
        

    def get_encoder_decoder(self, backbone: BackBone, loss_function: str = 'minimum_enclosing_ball_error', **kwargs):
        from nugets.models.encoder_decoders.shapefitting import MEBEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = input_dim + 1 # output is in the form [center, radius]
        return MEBEncoderDecoder(input_dim=input_dim, 
                                 backbone_input_dim = backbone_input_dim,
                                 backbone_output_dim = backbone_output_dim,
                                 output_dim = output_dim,
                                 loss_function=loss_function)


@register
class MinimumEnclosingAnnulusTask(ShapefittingTask):
    def label(self, pointset):
        from CGAL import CGAL_Bounding_volumes as cgal_bv
        import gc

        raw_pointset = pointset.numpy()
        flattened_pointcloud = raw_pointset.flatten().tolist()
        size = len(pointset)
        out = cgal_bv.min_annulus_d(flattened_pointcloud, size)
        label = [out[1][0], out[1][1], np.sqrt(out[0][0]), np.sqrt(out[0][1])]
        del out
        gc.collect()
        return torch.tensor(label).float()
    
    def compute_metrics(self, datapoint: LabeledSetDatapoint, results: Tensor):
        # TODO: improve metrics and logging here
        gt = datapoint.label
        gt_r = datapoint.label[:, -1]
        batch_radius_error = torch.abs(gt_r - results[:, -1])
        radius_relative_error = batch_radius_error/(gt_r + 1e-3)
        radius_mean_relative_error = radius_relative_error.mean()
        return dict(radius_relative_error=radius_mean_relative_error)
    
    def get_encoder_decoder(self, backbone: BackBone, loss_function: str='minimum_enclosing_annulus_error', **kwargs):
        from nugets.models.encoder_decoders.shapefitting import MEAEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        assert input_dim == 2 # TODO: Fix CGAL binding to work with > 2 dimensions
        output_dim = input_dim + 2
        return MEAEncoderDecoder(input_dim=input_dim, 
                                 backbone_input_dim=backbone_input_dim,
                                 backbone_output_dim=backbone_output_dim,
                                 output_dim=output_dim,
                                 loss_function=loss_function)

    
class MinimumCoveringEllipseTask(ShapefittingTask):
    def label(self, pointset):
        raise NotImplementedError
    
    def compute_metrics(self):
        raise NotImplementedError
    
    def get_encoder_decoder(self, backbone: BackBone, loss_function: str='minimum_enclosing_ellipse_error', **kwargs):
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = input_dim + 3 # extra dimensions for angle, major and minor radius. 
        return MEEEncoderDecoder(input_dim=input_dim, 
                                 backbone_input_dim = backbone_input_dim,
                                 backbone_output_dim = backbone_output_dim,
                                 output_dim = output_dim,
                                 loss_function=loss_function)


    
