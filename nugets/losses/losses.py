from typing import Any, Generic, dataclass_transform, get_type_hints, overload, Literal, TypeVar, Iterator, TYPE_CHECKING, ClassVar
from geomloss import SamplesLoss
import torch

class SinkhornLoss:
    """ Sinkhorn Loss function to use with point clouds """
    
    def __init__(self, **kwargs):
        self.loss = SamplesLoss(loss='sinkhorn', **kwargs)
    
    def __call__(self, set1, set2):
        return self.loss(set1.data.to(torch.float32), set2.data.to(torch.float32), ptr_x=set1.ptr, ptr_y=set2.ptr).mean()


