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

# Nearest neighbor: set of query points and point clouds. 
# Each point cloud gets processed via transformer/sumformer. 
# What is the datapoint: query vector, q, and point cloud, P.
# What is the architecture: 
# NN_1(P) -> returns representation of 

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

class SetQueryTask(Task):
    """
    Input: range/point cloud P, query object q
    Output: one-hot encoding detailing value in P that fulfills requirements of q. 
    """

    def process_dataset(self, dataset):
        transform = SetToQuerySetTransform(self.label)
        return transform(dataset)

    def datapoint_type(self):
        return QueryDatapoint

    def label(self, pointset):
        raise NotImplementedError

    def compute_metrics(self, datapoint: QueryDatapoint, results: Tensor):
        raise NotImplementedError
@register
class ExtremalPointTask(SetQueryTask):
    seed: int = 42 # Seed for randomly generated directions
    def label(self, pointset):
        import numpy as np
        rng = np.random.default_rng(self.seed)
        dim = self.dataset_info()["dim"]
        mean = np.ones(dim)
        cov = np.identity(dim)

        direction = rng.multivariate_normal(mean=mean,  cov=cov)
        direction = direction/np.linalg.norm(direction, ord=2)
        idx = np.argmax(pointset @ direction)
        label = np.zeros(pointset.size()[0])
        label[idx] = 1.0
        out = torch.tensor(label).unsqueeze(1)
        return pointset, out, torch.tensor(direction).type_as(pointset)
    
    def compute_metrics(self, datapoint:QueryDatapoint, results: Batch):
        return dict(BCE_loss=0.0)

    def get_encoder_decoder(self, backbone:BackBone, loss_function:str, **kwargs):
        from nugets.models.encoder_decoders.queries import SetQueryEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        input_dim = dataset_info["dim"]
        output_dim = 1
        return SetQueryEncoderDecoder(input_dim=input_dim, 
                                      backbone_input_dim=backbone_input_dim, 
                                      backbone_output_dim=backbone_output_dim, 
                                      output_dim=output_dim, 
                                      loss_function=loss_function)

@register
class NearestNeighborTask(SetQueryTask):
    seed: int = 42
    algorithm: 'auto' # We should be able to also use approximate NN here. 
    def label(self, pointset):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        raw_pointcloud = pointset.numpy()
        rng = np.random.default_rng(self.seed)
        q_idx = rng.choice(np.arange(len(pointset)))
        
        modified_pointset = np.delete(raw_pointcloud, q_idx, axis=0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm=algorithm).fit(modified_pointset)
        _, indices = nbrs.kneighbrs(raw_pointcloud[q_idx])
        nn_idx = indices[0][0]
        label = np.zeros(raw_pointcloud.shape[0])
        label[nn_idx] = 1.0
        out = torch.tensor(label).unsqueeze(1)

        return torch.tensor(modified_pointset), out.float(), pointset[q_idx]

    def get_encoder_decoder(self, backbone:BackBone, loss_function:str, **kwargs):
        
        raise NotImplementedError

