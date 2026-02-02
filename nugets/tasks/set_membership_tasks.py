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

from .task import Task
from .register import register
from .transforms import SetLabelTransform, SetToLabelSetTransform

class SetMembershipTask(Task):
    """
    Predict membership in a set
    """
    def process_dataset(self, dataset):
        transform = SetToLabelSetTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return SetToLabelSetDatapoint
    
    def label(self, pointset):
        raise NotImplementedError

    def compute_metrics(self, datapoint: SetToLabelSetDatapoint, results: Batch):
        # TODO: Add more logging metrics here
        membership = datapoint.labelset.data.squeeze(1)
        predicted_membership=results.data.squeeze(1)
        batch_index = datapoint.labelset.batch
        result = scatter_binary_cross_entropy(predicted=predicted_membership, target=membership, index=batch_index)
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
class ConvexHullMembershipTask(SetMembershipTask):
    """
    Predict membership in the convex hull. Returns a one-hot encoding corresponding to points in the convex hull. 
    """
    def label(self, pointset):
        from scipy.spatial import ConvexHull
        raw_pointcloud = pointset.numpy()
        out = ConvexHull(raw_pointcloud)
        label = np.zeros(raw_pointcloud.shape[0])
        label[out.vertices] = 1.0
        out = torch.tensor(label).unsqueeze(1)
        return out.float()
    
@register
class AlphaShape2MembershipTask(SetMembershipTask):
    """
    Predict membership in the convex hull. Returns a one-hot encoding corresponding to points in the convex hull. 
    """
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
