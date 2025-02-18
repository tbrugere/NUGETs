from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Transform, Datapoint
from torch import Tensor
from torch_heterogeneous_batching import Batch
from nugets.datasets.datapoint_types import DistanceDatapoint

import ot

from .task import Task
from .register import register



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
        transform = PairwiseDistances(self.distance)
        return transform(dataset)

    def distance(self, set1, set2):
        """Compute the distance between two sets"""
        raise NotImplementedError

    def get_encoder_decoder(self, backbone):
        """Get the encoder-decoder"""
        from nugets.models.encoder_decoders.distances import DistanceEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dims = backbone.get_input_dim()
        backbone_output_dim = backbone.get_output_dim()
        if "dim1" in dataset_info:
            input_dim1 = dataset_info["dim1"]
            input_dim2 = dataset_info["dim2"]
        else: 
            input_dim1 = input_dim2 = dataset_info["dim"]
        same_input_proj = getattr(backbone, 
                                  "same_input_proj", input_dim1 == input_dim2)
        return DistanceEncoderDecoder(input_dim = (input_dim1, input_dim2), 
                                      backbone_input_dim=backbone_input_dims, 
                                      backbone_output_dim=backbone_output_dim, 
                                      same_input_proj=same_input_proj)

    def datapoint_type(self):
        return DistanceDatapoint

@register
class WassersteinDistanceTask(DistanceTask):
    def distance(self, set1, set2, p=1, sinkhorn=0):
        """Compute the p-Wasserstein distance between two sets"""

        M = ot.dist(set1, set2, metric='euclidean')**p
        sz_a = len(set1)
        sz_b = len(set2)
        a = np.ones(sz_a)/sz_a
        b = np.ones(sz_b)/sz_b
        if sinkhorn > 0:
            return ot.sinkhorn2(a, b, M, reg=sinkhorn)
        return ot.emd2(a, b, M)**(1/p)
