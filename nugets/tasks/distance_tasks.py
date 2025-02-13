from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Transform, Datapoint
from torch import Tensor
from torch_heterogeneous_batching import Batch

import ot

from .task import Task

@dataclass
class DistanceDatapoint(Datapoint):
    set1: Tensor
    set2: Tensor
    distance: Tensor

    @staticmethod
    def collate(batch):
        set1 = Batch.from_list([x.set1 for x in batch])
        set2 = Batch.from_list([x.set2 for x in batch])
        distance = torch.stack([x.distance for x in batch])
        return DistanceBatch(set1=set1, set2=set2, distance=distance)

@dataclass
class DistanceBatch(Datapoint):
    set1: Batch
    set2: Batch
    distance: Tensor

def triangular_index(i):
    r"""Decomposes i into (k, l) where k < l

    so that 
    .. math::

        i = 0 + 1 + 2 + ... + (l - 1) + k

    ie 
    .. math::

        i = frac{l * (l - 1)}{2} + k
    """
    # l = (1 + sqrt(1 + 8 * i) / 2) 
    l = int((sqrt(1 + 8 * i) - 1) / 2) + 1

    k = i - l * (l - 1) // 2
    return k, l

class PairwiseDistances(Transform):

    distance: Callable[[Tensor, Tensor], Tensor]

    def __init__(self, distance):
        self.distance = distance

    def __len__(self):
        return len(self.inner) * (len(self.inner) - 1) // 2

    def __getitem__(self, idx):
        k, l = triangular_index(idx)
        set1 = self.inner[k]
        set2 = self.inner[l]
        distance = self.distance(set1, set2)
        return DistanceDatapoint(set1=set1, set2=set2, distance=distance)


class DistanceTask(Task):
    """Task for learning to predict a distance between two sets"""

    def process_dataset(self, dataset):
        transform = Transform(self.distance)
        return transform(dataset)

    def distance(self, set1, set2):
        """Compute the distance between two sets"""
        raise NotImplementedError

    def get_encoder_decoder(self, backbone):
        """Get the encoder-decoder"""
        from nugets.models.encoder_decoders.distances import DistanceEncoderDecoder
        return DistanceEncoderDecoder(backbone)

    def datapoint_type(self):
        return DistanceDatapoint

class WassersteinDistanceTask(DistanceTask):
    def distance(self, set1, set2, p=1, sinkhorn=0):
        """Compute the p-Wasserstein distance between two sets"""
        if sinkhorn > 0:
            ot.sinkhorn2()
        ot.emd2()
        return 0
