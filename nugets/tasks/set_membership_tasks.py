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
from torch_scatter import scatter
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint, SetToLabelSetBatch, SetToLabelSetDatapoint
from nugets.models.backbone import BackBone
from nugets.losses.losses import scatter_binary_cross_entropy

from .task import Task
from .register import register
from .transforms import SetLabelTransform, SetToLabelSetTransform

class SetMembershipTask(Task):
    """
    Predict membership in a set.
    Used as the base class for ConvexHulls and AlphaShapes
    """
    def process_dataset(self, dataset):
        transform = SetToLabelSetTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return SetToLabelSetDatapoint
    
    def label(self, pointset):
        raise NotImplementedError

    def compute_metrics(self, datapoint: SetToLabelSetDatapoint, results: Batch):
        """
        Tracking accuracy and binary cross entropy loss
        """
        
        membership = datapoint.labelset.data.squeeze(1)
        predicted_membership_logits=results.data.squeeze(1)
        batch_index = datapoint.labelset.batch

        # Average accuracy per point cloud
        pm_probs = torch.sigmoid(predicted_membership_logits)
        predicted_membership = (pm_probs >=0.5).int()
        membership = membership.int()
        correct = (predicted_membership == membership).int()
        accuracy = scatter(src=correct, index=batch_index, reduce="mean")

        #precision
        tn_mask = membership == 0
        tp_mask = membership == 1
        sum_tps = scatter(src=(predicted_membership[tp_mask] == 1).int(), index=batch_index[tp_mask], reduce='sum')
        sum_fps = scatter(src=(predicted_membership[tn_mask] == 1).int(), index=batch_index[tn_mask], reduce='sum')
        precision_per_cloud = sum_tps/(sum_tps + sum_fps + 1e-5)

        # Recall
        tp_mask = membership == 1
        sum_fns = scatter(src=(predicted_membership[tp_mask] == 0).int(), index=batch_index[tp_mask], reduce='sum')

        recall_per_cloud = sum_tps/(sum_tps + sum_fns)
        # f1 score
        f1 = 2 * sum_tps/(2*sum_tps + sum_fps + sum_fns + 1e-5)
        return dict( accuracy=torch.mean(accuracy, dtype=torch.float32), 
                     recall=torch.mean(recall_per_cloud, dtype=torch.float32),
                     precision=torch.mean(precision_per_cloud, dtype=torch.float32), 
                     f1 = torch.mean(f1, dtype=torch.float32))
    
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
    def __init__(self, *args, alpha=5.0, use_cgal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.use_cgal=use_cgal            
        
    def label(self, pointset):
        """
        Use CGAL to compute an alpha shape for 2D points
        """
        alpha = self.alpha
        use_cgal= self.use_cgal
        if use_cgal:
            from CGAL.CGAL_Kernel import Point_2
            from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2
            from CGAL.CGAL_Alpha_shape_2 import Weighted_alpha_shape_2_Face_handle
            from CGAL.CGAL_Alpha_shape_2 import GENERAL, EXTERIOR, SINGULAR, REGULAR, INTERIOR
            from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2_Vertex_handle

            # TODO: Add assertion for 2D alpha shape
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
        else:
            edges, mask = alpha_shape_2d(pointset, alpha=alpha)
            mask = np.array(mask, dtype=np.float32)
            out = torch.tensor(mask).unsqueeze(1)
            return out.float()



def alpha_shape_2d(points, alpha=5.0):
    from math import isfinite, sqrt
    
    """
    Return (boundary_segments, boundary_mask), using squared-radius alpha.

    Python only implementation of AlphaShapes. O(n^3) time; suitable for small point clouds.
    """
    original = [tuple(map(float, p)) for p in points]
    alpha = float(alpha)
    
    if not isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if any(len(p) != 2 or not all(map(isfinite, p)) for p in original):
        raise ValueError("Expected finite 2D points")

    points = list(dict.fromkeys(original))  # Handle duplicate coordinates.
    edges = []
    connected = set()
    boundary = set()
    tolerance = 1e-12 * alpha

    for i, (ax, ay) in enumerate(points):
        for j in range(i + 1, len(points)):
            bx, by = points[j]
            dx, dy = bx - ax, by - ay
            distance_sq = dx * dx + dy * dy

            if distance_sq > 4 * alpha:
                continue

            connected.update((i, j))
            mx, my = ax + dx / 2, ay + dy / 2
            offset = sqrt(max(0.0, alpha / distance_sq - 0.25))

            # Two possible centers for a radius-sqrt(alpha) circle.
            for sign in (-1, 1):
                cx = mx - sign * dy * offset
                cy = my + sign * dx * offset
                empty = all(
                    (x - cx) ** 2 + (y - cy) ** 2 >= alpha - tolerance
                    for k, (x, y) in enumerate(points)
                    if k != i and k != j
                )
                if empty:
                    edges.append((points[i], points[j]))
                    boundary.update((i, j))
                    break

    # A point is isolated when no other point is within 2*sqrt(alpha).
    boundary.update(set(range(len(points))) - connected)
    boundary_points = {points[i] for i in boundary}
    mask = [p in boundary_points for p in original]
    return edges, mask