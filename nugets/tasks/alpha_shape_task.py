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

@register
class AlphaShape2Task(Task):
    def process_dataset(self, dataset):
        transform = SetToLabelSetTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return SetToLabelSetDatapoint

    def label(self, pointset, alpha=5.0):
        """
        Use CGAL to compute an alpha shape for 2D points
        """
        from CGAL.CGAL_Kernel import Point_2
        from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2
        from CGAL.CGAL_Alpha_shape_2 import Weighted_alpha_shape_2_Face_handle
        from CGAL.CGAL_Alpha_shape_2 import GENERAL, EXTERIOR, SINGULAR, REGULAR, INTERIOR
        from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2_Vertex_handle

        # TODO: Add assertion for 2D alpha shape
        # TODO: Add task parameters to set the alpha in alpha shape
        cgal_pts = []
        for pt in pointset:
            cgal_pts.append(Point_2(pt[0].item(), pt[1].item()))
        t = Alpha_shape_2(cgal_pts, alpha, GENERAL)
        t.clear()
        t.make_alpha_shape(cgal_pts)
        t.set_alpha(alpha)
        n_points = len(cgal_pts)
        in_shape = np.zeros(n_points)
        for i in range(len(cgal_pts)):
            v = cgal_pts[i]
            vert_type = t.classify(v)
            if vert_type in [SINGULAR, REGULAR]:
                in_shape[i] = 1.0
        out = torch.tensor(in_shape).unsqueeze(1)
        return out.float()
                
    
    def compute_metrics(self, datapoint, results):
        # TODO: Add more logging metrics here
        alpha_shape = datapoint.labelset.data.squeeze(1)
        predicted_membership=results.data.squeeze(1)
        batch_index = datapoint.labelset.batch
        result = scatter_binary_cross_entropy(predicted=predicted_membership, target=alpha_shape, index=batch_index)
        return dict(BCE_loss=result)
    
    def get_encoder_decoder(self, backbone: BackBone):
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