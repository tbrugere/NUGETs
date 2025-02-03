from dataclasses import dataclass
from typing import Literal
import torch
import numpy as np

from ml_lib.datasets import Datapoint, register as dataset_register
from ml_lib.datasets.datasets.randomly_generated_dataset import GeneratedDataset
from torch_heterogeneous_batching.batch import Batch

@dataclass
class Set_datapoint(Datapoint):
    pointset: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_sets = [p.pointset for p in points]
        set_batch = Batch.from_list(input_sets, order=1)
        return Set_batch(set_batch)

@dataclass
class Set_batch(Datapoint):
    pointset: Batch

@dataclass
class Point_datapoint(Datapoint):
    point: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_points = [p.point for p in points]
        return cls(torch.stack(input_points, dim=0))


