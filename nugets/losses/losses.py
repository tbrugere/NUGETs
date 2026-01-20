from typing import Any, Generic, dataclass_transform, get_type_hints, overload, Literal, TypeVar, Iterator, TYPE_CHECKING, ClassVar
from geomloss import SamplesLoss
import torch
from torch.nn.functional import mse_loss, l1_loss

class SinkhornLoss:
    """ Sinkhorn Loss function to use with point clouds """
    
    def __init__(self, **kwargs):
        self.loss = SamplesLoss(loss='sinkhorn', **kwargs)
    
    def __call__(self, set1, set2):
        return self.loss(set1.data.to(torch.float32), set2.data.to(torch.float32), ptr_x=set1.ptr, ptr_y=set2.ptr).mean()


def minimum_enclosing_ball_error(c, predicted_c, r, predicted_r, p=2, **kwargs):
    """
    loss function for minimum enclosing ball:
    minimize the L_p distance between the center and predicted center and the difference in radii
    """
    err = torch.mean(torch.linalg.norm(c - predicted_c, axis=1)) + mse_loss(r, predicted_r)
    return err 

def radius_error(r, predicted_r, **kwargs):
    err = mse_loss(r, predicted_r)
    return err