from typing import Literal
import numpy as np
import torch

from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from nugets.datasets.register import register as dataset_register
from .datapoint_types import Point_datapoint, Set_datapoint


################################## set-level datasets
@dataset_register 
class USGSTerrains(Dataset[Set_datapoint]):
    datatype = Set_datapoint
    min_points: int
    max_points: int

    def __init__(self, dim: int=2, min_points=8, max_points=32, **kwargs):
        super().__init__(**kwargs)
        self.dim=3 
        self.min_points = min_points
        self.max_points = max_points
        self.mean = np.ones(dim)
        self.covariance = np.identity(dim)
    
    def prepare(self):
        pass
    
    def generate_item(self, rng):
        n_points = rng.integers(self.min_points, self.max_points)
        points = rng.multivariate_normal(mean=self.mean, cov=self.covariance, size=n_points)
        return Set_datapoint(pointset=torch.as_tensor(points, dtype=torch.float32))
    
    def dataset_parameters(self):
        return {'dim': self.dim, 'min_points': self.min_points, 'max_points': self.max_points}
