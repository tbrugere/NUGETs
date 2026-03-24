from typing import Literal
import numpy as np
import torch
from ml_lib.datasets.datasets.randomly_generated_dataset import GeneratedDataset

from nugets.datasets.register import register as dataset_register
from .datapoint_types import Point_datapoint, Set_datapoint

from scipy.stats import qmc
import warnings
from polygenerator import random_polygon
from yupi.generators import RandomWalkGenerator, LangevinGenerator

@dataset_register
class RandomTrajectory(GeneratedDataset[Set_datapoint]):
    """
    Generate a set of points representing a random walk. The order 
    of the points indicates the path of the walk. 
    """
    datatype = Set_datapoint
    dim: int
    min_points: int
    max_points: int
    scaling: float
    type: str

    def __init__(self, dim: int = 2, min_points: int=5, max_points: int=12, scaling: float=1.0, type: str = 'RandomWalk', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.min_points = min_points
        self.max_points = max_points
        self.scaling = scaling
        self.dt=1
        match type: 
            case 'RandomWalk':
                self.trajectory_generator = RandomWalkGenerator
            case 'Langevin':
                self.trajectory_generator = LangevinGenerator
            case other:
                warnings.warn("Unrecognized trajectory generator, defaulting to random walk.")
                self.trajectory_generator = RandomWalkGenerator

    def generate_item(self, rng):
        n_points = rng.integers(self.min_points, self.max_points)
        gen = self.trajectory_generator(T=n_points, dim=self.dim, dt=self.dt)
        traj = gen.generate(1)[0]
        trajectory = np.array(traj.r)

        # center item
        trajectory = trajectory - np.mean(trajectory, axis=0)
        return Set_datapoint(pointset=torch.as_tensor(trajectory, dtype=torch.float32))
    
    def dataset_parameters(self):
        return {'dim': self.dim, 'min_points': self.min_points, 'max_points': self.max_points, 'bound': self.scaling}
