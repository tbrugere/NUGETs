from typing import Any, Generic, dataclass_transform, get_type_hints, overload, Literal, TypeVar, Iterator, TYPE_CHECKING, ClassVar
from geomloss import SamplesLoss
import torch
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy_with_logits
from torch_scatter import scatter

import sys

class SinkhornLoss:
    """ Sinkhorn Loss function to use with point clouds """
    
    def __init__(self, **kwargs):
        self.loss = SamplesLoss(loss='sinkhorn', **kwargs)
    
    def __call__(self, set1, set2):
        return self.loss(set1.data.to(torch.float32), set2.data.to(torch.float32), ptr_x=set1.ptr, ptr_y=set2.ptr).mean()

def minimum_ball_error(c, predicted_c, r, predicted_r, p=2, **kwargs):
    """
    loss function for minimum enclosing ball:
    minimize the L_p distance between the center and predicted center and the difference in radii
    """
    err = torch.mean(torch.linalg.norm(c - predicted_c, axis=1)) + mse_loss(r, predicted_r)
    return err 

def minimum_ball_radius_error(r, predicted_r, **kwargs):
    err = mse_loss(r, predicted_r)
    return err

def minimum_annulus_error(c, predicted_c, inner_r, predicted_inner_r, outer_r, predicted_outer_r, **kwargs):
    err = torch.mean(torch.linalg.norm(c - predicted_c, axis=1)) + minimum_annulus_radius_error(inner_r, predicted_inner_r, outer_r, predicted_outer_r)
    return err

def minimum_annulus_radius_error(inner_r, predicted_inner_r, outer_r, predicted_outer_r, **kwargs):
    inner_r_err = mse_loss(inner_r, predicted_inner_r)
    outer_r_err = mse_loss(outer_r, predicted_outer_r)
    return inner_r_err + outer_r_err

def minimum_enclosing_ellipse_error(**kwargs):
    raise NotImplementedError


def scatter_binary_cross_entropy(predicted, target, index, reduction="mean", **kwargs):
    " Batch compatible cross entropy loss "
    unrolled = binary_cross_entropy_with_logits(input=predicted, target=target, reduction="none")
    per_set_bce_error=scatter(src=unrolled, index=index, reduce=reduction)
    return per_set_bce_error.mean()

def binary_focal_loss(inputs, targets, gamma=2):
    """ Focal loss for binary classification. """
    probs = torch.sigmoid(inputs)
    targets = targets.float()

    # Compute binary cross entropy
    bce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    # Compute focal weight
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    loss = focal_weight * bce_loss

    return loss

def scatter_binary_focal_loss(predicted, target, index, reduction='mean', **kwargs):
    unrolled = binary_focal_loss(inputs=predicted, targets=target)
    per_set_fl_error = scatter(src=unrolled, index=index, reduce=reduction)
    return per_set_fl_error.mean()