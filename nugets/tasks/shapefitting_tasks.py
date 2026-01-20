from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Transform, Datapoint
# import numpy as np
import ot
from scipy.spatial.distance import directed_hausdorff
import torch
from torch import Tensor
from torch_heterogeneous_batching import Batch
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint
from nugets.models.backbone import BackBone

from .task import Task
from .register import register
import miniball
import numpy as np
#from CGAL import CGAL_Bounding_volumes as cgal_bv

class ShapefittingTransform(Transform):
    label: Callable[Tensor, Tensor]
    def __init__(self, label, normalization='identity'):
        self.label = label

    def __len__(self):
        return len(self.inner)
    
    def __getitem__(self, idx):
        pointset = self.inner[idx].pointset
        label = self.label(pointset)
        return LabeledSetDatapoint(pointset=pointset, label=label)

class ShapefittingTask(Task):
    """Task for fitting a shape to a set"""
    def process_dataset(self, dataset):
        transform = ShapefittingTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return LabeledSetDatapoint
    
    def label(self, pointset):
        raise NotImplementedError

@register
class MinimumEnclosingBallTask(ShapefittingTask):
    def label(self, pointset):
        raw_pointcloud = pointset.numpy()
        C, r = miniball.get_bounding_ball(raw_pointcloud, epsilon=1e-6)
        return torch.tensor(np.append(C, r)).float()
    
    def compute_metrics(self, datapoint: LabeledSetDatapoint, results: Tensor):
        gt = datapoint.label
        gt_r = datapoint.label[:, -1]
        batch_radius_error = torch.abs(gt_r - results[:, -1])
        radius_relative_error = batch_radius_error/(gt_r + 1e-3)
        radius_mean_relative_error = radius_relative_error.mean()
        return dict(radius_relative_error=radius_mean_relative_error)
        

    def get_encoder_decoder(self, backbone: BackBone, loss_function: str = 'mse_loss', **kwargs):
        from nugets.models.encoder_decoders.shapefitting import MEBEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = input_dim + 1
        return MEBEncoderDecoder(input_dim=input_dim, 
                                 backbone_input_dim = backbone_input_dim,
                                 backbone_output_dim = backbone_output_dim,
                                 output_dim = output_dim,
                                 loss_function=loss_function)


class MinimumAnnulusTask(Task):
    """Task for learning to predict the center, inner radius, and outer radius of minimum enclosing annulus"""
    def get_encoder_decoder(self, backbone):
        """ 

        Get the encoder-decoder for shape fitting. 
        Note that the output dimension for the decoder will be the input dimension of the encoder + 2,
        where the last two dimensions are used to store the inner radius and outer radius.
        TODO: Fix CGAL binding so this works for more then 2D points

        """
        from nugets.models.encoder_decoders.shapefitting import MEAEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        assert input_dim == 2 # This can be taken out once CGAL binding is fixed
        output_dim = input_dim + 2
        return MEAEncoderDecoder(input_dim=input_dim, 
                                 backbone_input_dim = backbone_input_dim,
                                 backbone_output_dim = backbone_output_dim,
                                 output_dim = output_dim)
    
    def get_minimum_enclosing_annulus(self, input_set):
        input_pts = input_set.tolist() ## syntax issue likely
        return cgal_bv.min_annulus_d(input_pts, len(input_pts))

    
class MinimumCoveringEllipseTask(Task):
    def get_encoder_decoder(self, backbone):
        raise NotImplementedError
    def get_minimum_covering_ellipse(self, input):
        return NotImplementedError
    
