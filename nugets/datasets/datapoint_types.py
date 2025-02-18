from dataclasses import dataclass
from typing import Literal
import torch
from torch import Tensor
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


@dataclass
class DistanceDatapoint(Datapoint):
    set1: Tensor
    set2: Tensor
    distance: Tensor

    @staticmethod
    def collate(batch):
        set1 = Batch.from_list([x.set1 for x in batch], order=1)
        set2 = Batch.from_list([x.set2 for x in batch], order=1)
        distance = torch.stack([x.distance for x in batch])
        return DistanceBatch(set1=set1, set2=set2, distance=distance)

@dataclass
class DistanceBatch(Datapoint):
    set1: Batch
    set2: Batch
    distance: Tensor
