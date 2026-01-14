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
from nugets.datasets.datapoint_types import DistanceDatapoint

from .task import Task
from .register import register
#from CGAL import CGAL_Bounding_volumes as cgal_bv


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


class MinimumBallTask(Task):
    """"Task for learning to predict the center and radius of a minimum enclosing ball"""

    def get_encoder_decoder(self, backbone):
        from nugets.model.encoder_decoders.shapefitting import MEBEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = input_dim + 1
        return MEAEncoderDecoder(input_dim=input_dim, 
                                 backbone_input_dim = backbone_input_dim,
                                 backbone_output_dim = backbone_output_dim,
                                 output_dim = output_dim)
    
    def get_minimum_enclosing_ball(self, set):
        return NotImplementedError

class MinimumCoveringEllipseTask(Task):
    def get_encoder_decoder(self, backbone):
        raise NotImplementedError
    def get_minimum_covering_ellipse(self, input):
        return NotImplementedError
    
