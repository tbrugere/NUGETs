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
from nugets.datasets.datapoint_types import QueryDatapoint, QueryBatch
from nugets.models.backbone import BackBone

from .task import Task
from .register import register
from .transforms import SetLabelTransform

from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_max
from scipy.stats import qmc

import shapely
import numpy as np
import sys 

class SetToQuerySetTransform(Transform):
    label: Callable[Tensor, Tensor]
    def __init__(self, label):
        self.label=label

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        pointset=self.inner[idx].pointset
        pointset, label, query =self.label(pointset)
        return QueryDatapoint(pointset=pointset, label=label, query=query)

class SetToPointRegressionTask(Task):
    """
    Base task for tasks where one needs the index. 
    """
    def process_dataset(self, dataset):
        transform = SetToQuerySetTransform(self.label)
        return transform(dataset)

    def label(self, pointset):
        raise NotImplementedError

    def datapoint_type(self):
        return QueryDatapoint

    def compute_metrics(self, datapoint: QueryDatapoint, results: Tensor):
        probs = softmax(src=results, index=datapoint.pointset.batch)
        scaled_pts = probs.unsqueeze(1)*datapoint.pointset.data
        nn_pred = scatter(src=scaled_pts, index=datapoint.pointset.batch, reduce='sum', dim=0)
        err = torch.norm(nn_pred - datapoint.label, dim=1)
        err_avg = err.mean()
        
        # hardmax neighbor error (predicted nearest neighbor, not convex combination of original point set)
        _, argmax_idx = scatter_max(src=probs, index=datapoint.pointset.batch)
        nn_pred_hardmax = datapoint.pointset.data[argmax_idx]
        hardmax_err = torch.norm(nn_pred_hardmax - datapoint.label, dim=1)
        hardmax_err_avg = hardmax_err.mean()
        return dict(softmax_neighbor_error=err_avg, hardmax_neighbor_error=hardmax_err_avg)

    def get_encoder_decoder(self, backbone: BackBone, loss_function: str, **kwargs):
        from nugets.models.encoder_decoders.queries import SetToPointRegressionEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = backbone_output_dim
        output_dim = input_dim
        return SetToPointRegressionEncoderDecoder(input_dim=input_dim, 
                                                 backbone_input_dim=backbone_input_dim, 
                                                 backbone_output_dim=backbone_output_dim, 
                                                 output_dim=output_dim, 
                                                 loss_function=loss_function)

@register
class ApproximateQueryRegressionTask(Task):
    """
    Task for outputting the result of the query.
    For tasks like nearest neighbors and extremal points: the output is not guaranteed to be a member of the original point cloud.
    (i.e. the encoder/decoder does not output a set of logits but rather an estimate for the nearest neighbor directly). 
    However, this class can be used to output logits for cases like range queries. 
    """

    def process_dataset(self, dataset):
        transform = SetToQuerySetTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return QueryDatapoint

    def label(self, pointset):
        raise NotImplementedError

    def compute_metrics(self, pointset):
        raise NotImplementedError
    
    def get_encoder_decoder(self,  backbone: BackBone, loss_function: str, **kwargs):
        from nugets.models.encoder_decoders.queries import ApproximateQueryEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = input_dim
        return ApproximateQueryEncoderDecoder(input_dim=input_dim, 
                                              backbone_input_dim=backbone_input_dim, 
                                              backbone_output_dim=backbone_output_dim, 
                                              output_dim=output_dim, 
                                              loss_function=loss_function)


@register
class ApproximateExtremalPointTask(ApproximateQueryRegressionTask): 
    seed: int=42
    rng = np.random.default_rng(seed)
    def label(self, pointset):
        dim = self.dataset_info()["dim"]
        mean = np.ones(dim)
        cov = np.identity(dim)
        direction = self.rng.multivariate_normal(mean=mean,  cov=cov)
        direction = direction/np.linalg.norm(direction, ord=2)
        idx = np.argmax(pointset @ direction)
        label = pointset[idx]
        return pointset, torch.tensor(label).float(), torch.tensor(direction).type_as(pointset)

@register
class ApproximateNearestNeighborTask(ApproximateQueryRegressionTask):
    seed: int = 42
    algorithm: str = 'auto'
    rng = np.random.default_rng(seed)
    def label(self, pointset):
        from sklearn.neighbors import NearestNeighbors
        raw_pointcloud = pointset.numpy()
        q_idx = self.rng.choice(np.arange(len(pointset)))
        
        modified_pointset = np.delete(raw_pointcloud, q_idx, axis=0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm).fit(modified_pointset)
        _, indices = nbrs.kneighbors([raw_pointcloud[q_idx]])
        nn_idx = indices[0][0]
        return torch.tensor(modified_pointset), torch.tensor(raw_pointcloud[nn_idx]).float(), pointset[q_idx]


@register
class NearestNeighborRegressionTask(SetToPointRegressionTask):
    """
    Task for learning to output the nearest neighbor given an input point cloud and query.
    Training examples are in the form (S, l, q) where S is the point cloud, l is the "label" (or the nearest neighbor)
    and q is the query vector. 
    The output of the encoder/deocder is a set of logits. 
    """
    seed: int=42 
    algorithm: str = 'auto'
    rng = np.random.default_rng(seed)
    
    def label(self, pointset):
        from sklearn.neighbors import NearestNeighbors
        raw_pointcloud = pointset.numpy()
        q_idx = self.rng.choice(np.arange(len(pointset)))
        
        modified_pointset = np.delete(raw_pointcloud, q_idx, axis=0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm).fit(modified_pointset)
        _, indices = nbrs.kneighbors([raw_pointcloud[q_idx]])
        nn_idx = indices[0][0]
        return torch.tensor(modified_pointset), torch.tensor(raw_pointcloud[nn_idx]).float(), pointset[q_idx]

@register
class ExtremalPointRegressionTask(SetToPointRegressionTask):
    """
    Task for learning to output the furthest point in a direction given an input point cloud and a query direction.
    Training examples are in the form (S, l, q) where S is the point cloud, l is the "label" (or the furthest point)
    and q is the query vector. 
    The output of the encoder/deocder is a set of logits. 
    """
    seed: int=42
    rng = np.random.default_rng(seed)
    def label(self, pointset):
        dim = self.dataset_info()["dim"]
        mean = np.ones(dim)
        cov = np.identity(dim)
        direction = self.rng.multivariate_normal(mean=mean,  cov=cov)
        direction = direction/np.linalg.norm(direction, ord=2)
        idx = np.argmax(pointset @ direction)
        label = pointset[idx]
        return pointset, torch.tensor(label).float(), torch.tensor(direction).type_as(pointset)

@register 
class RangeQueryTask(Task):
    """
    Given a set of points representing a polygon and a query point, output
    whether or not the query point is inside the polygon. 
    """

    seed: int = 42
    query_sampling: str = 'random'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bound = self.dataset_info()['bound'] + 1
        self.dim = self.dataset_info()['dim']
        self.sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)
        self.rng = np.random.default_rng(seed=self.seed)

    def process_dataset(self, dataset):
        transform = SetToQuerySetTransform(self.label)
        return transform(dataset)
    
    def datapoint_type(self):
        return QueryDatapoint
    
    def label(self, pointset):
        v = self.rng.uniform(low=0.0, high=1.0)
        in_bound = v > 0.5
        area = shapely.Polygon(pointset.numpy())
        if in_bound:
            pt = shapely.point_on_surface(area)
            x = self.rng.normal(scale=0.5)
            y = self.rng.normal(scale=0.5)
            query = np.array([pt.x + x, pt.y + y])
        else:
            query = self.sampler.random(n=1)
            query = qmc.scale(query, [-self.bound, -self.bound], [self.bound, self.bound])[0]
        area = shapely.Polygon(pointset.numpy())
        query_point = shapely.Point(query)
        val = shapely.contains(area, query_point)
        return pointset, torch.tensor(int(val)), torch.tensor(query).float()
    
    def compute_metrics(self, datapoint: QueryDatapoint, results: Tensor):
        ground_truth = datapoint.label
        output = torch.sigmoid(results)
        predictions = (output >=0.5).int()
        # accuracy 
        correct = (predictions == ground_truth).int()
        accuracy = torch.mean(correct.float())
        # precision
        tp_mask = ground_truth == 1
        tn_mask = ground_truth == 0
        sum_tps = torch.sum((predictions[tp_mask] == 1).int())
        sum_fps = torch.sum((predictions[tn_mask] == 1).int())
        precision = sum_tps/(sum_tps+sum_fps + 1e-5)

        # Recall
        sum_fns = torch.sum((predictions[tp_mask] == 0).int())
        recall = sum_tps/(sum_tps + sum_fns + 1e-5)
        # f1 score
        f1 = (2* recall * precision) / (precision+ recall + 1e-5)
        return dict(accuracy = accuracy, f1=f1, recall=recall, precision=precision)
    
    def get_encoder_decoder(self, backbone: BackBone, loss_function: str, **kwargs):
        from nugets.models.encoder_decoders.queries import ApproximateQueryEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = 1
        return ApproximateQueryEncoderDecoder(input_dim=input_dim, 
                                                backbone_input_dim=backbone_input_dim, 
                                                backbone_output_dim=backbone_output_dim, 
                                                output_dim=output_dim, 
                                                loss_function=loss_function)
