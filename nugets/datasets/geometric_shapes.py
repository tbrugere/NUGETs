from typing import Literal
import numpy as np
import torch
from ml_lib.datasets.datasets.randomly_generated_dataset import GeneratedDataset

from nugets.datasets.register import register as dataset_register
from .datapoint_types import Point_datapoint, Set_datapoint

from scipy.stats import qmc


################################## point-level datasets
@dataset_register
class Gaussian(GeneratedDataset[Point_datapoint]):
    """Gaussian dataset

    Dataset of points sampled from a 2D Gaussian distribution.
    """
    datatype = Point_datapoint
    mean: tuple[float, float]
    std: tuple[float, float]
    dim: int

    def __init__(self, *, mean=(0, 0), std=(1, 1), dim=2, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.dim = dim

    def prepare(self):
        pass

    def generate_item(self, rng):
        point = rng.normal(self.mean, self.std, self.dim)
        return Point_datapoint(point=torch.as_tensor(point, dtype=torch.float32))

    def dataset_parameters(self):
        return {'mean': self.mean, 'std': self.std, 'dim': self.dim}

@dataset_register
class Torus4D(GeneratedDataset[Point_datapoint]):
    """4D Torus dataset

    Dataset of points sampled from a 4D torus.
    """
    datatype = Point_datapoint

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare(self):
        pass

    def generate_item(self, rng):
        u = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0, 2 * np.pi)
        point = np.array([np.cos(u), np.sin(u), np.cos(v), np.sin(v)])
        return Point_datapoint(point=torch.as_tensor(point, dtype=torch.float32))

    def dataset_parameters(self):
        return {'dim': 2, }



################################## set-level datasets
@dataset_register 
class GaussianBlobs(GeneratedDataset[Set_datapoint]):
    datatype = Set_datapoint
    dim: int
    min_points: int
    max_points: int
    centers: int
    seed: int = 42
    bound: int = 10

    def __init__(self, dim: int=2, min_points=8, max_points=32, centers=3, **kwargs):
        super().__init__(**kwargs)
        self.dim=dim 
        self.min_points = min_points
        self.max_points = max_points
        self.mean = np.ones(dim)
        self.covariance = np.identity(dim)
        self.centers = centers
        self.center_sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)    

    def prepare(self):
        pass
    
    def generate_item(self, rng):
        unit_sample = self.center_sampler.random(n=self.centers)
        sample_means = qmc.scale(unit_sample, [-self.bound, -self.bound], [self.bound, self.bound])

        n_points = rng.integers(self.min_points, self.max_points)
        cuts = np.random.choice(np.arange(1, n_points), self.centers - 1, replace=False)
        cuts = np.sort(cuts)
        cuts = np.concatenate(([0], cuts, [n_points]))
        partitions = np.diff(cuts)
        
        points_ = []
        for i in range(self.centers):
            center = sample_means[i]
            p_size = partitions[i]
            pts = rng.multivariate_normal(mean=center, cov=self.covariance, size=p_size)
            points_.append(pts)
        points = np.concatenate(points_)
        # points = rng.multivariate_normal(mean=self.mean, cov=self.covariance, size=n_points)
        return Set_datapoint(pointset=torch.as_tensor(points, dtype=torch.float32))
    
    def dataset_parameters(self):
        return {'dim': self.dim, 'min_points': self.min_points, 'max_points': self.max_points}

@dataset_register
class GrowingCircles(GeneratedDataset[Set_datapoint]):
    """Circles whose radius grow with the number of points

    Dataset of sets of points disposed sampled uniformly on a circle.
    The circles are centered at the origin and have 
    a radius that grows with the number of points.
    """
    datatype = Set_datapoint

    dim: int
    min_points: int
    max_points: int
    size_multiplier: float
    radius: Literal['linear', 'sqrt', 'log', 'constant']

    def __init__(self, dim: int = 2, min_points=8, max_points=32, size_multiplier=1.,
                 radius: Literal['linear', 'sqrt', 'log', 'constant']="linear", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.min_points = min_points
        self.max_points = max_points
        self.size_multiplier = size_multiplier
        self.radius = radius

    def prepare(self):
        pass

    def generate_item(self, rng):
        n_points = rng.integers(self.min_points, self.max_points)
        angles = rng.random((n_points, 1)) * 2 * np.pi
        points = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        radius = self.compute_radius(n_points)
        points = points * radius
        return Set_datapoint(pointset=torch.as_tensor(points, dtype=torch.float32))

    def compute_radius(self, n_points):
        f = {'linear': lambda x: x, 'sqrt': np.sqrt, 'log': np.log2, 'constant': lambda _: 1}
        return f[self.radius](n_points) * self.size_multiplier

    def dataset_parameters(self):
        return {'dim': self.dim, 'min_points': self.min_points, 'max_points': self.max_points}
