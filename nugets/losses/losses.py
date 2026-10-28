"""Loss functions for geometric prediction and point-cloud classification."""

from typing import Any, Generic, dataclass_transform, get_type_hints, overload, Literal, TypeVar, Iterator, TYPE_CHECKING, ClassVar
from geomloss import SamplesLoss
import torch
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy_with_logits
from torch_scatter import scatter
from scipy.stats import uniform_direction

import sys

class SinkhornLoss:
    """Compute the mean Sinkhorn loss between batches of point clouds.

    :param kwargs: Keyword arguments forwarded to ``geomloss.SamplesLoss``
        with ``loss='sinkhorn'``.
    """
    
    def __init__(self, **kwargs):
        self.loss = SamplesLoss(loss='sinkhorn', **kwargs)
    
    def __call__(self, set1, set2):
        """Evaluate the loss for two batches of point clouds.

        :param set1: First batch, with point coordinates in ``data`` and
            batch pointers in ``ptr``.
        :param set2: Second batch, with point coordinates in ``data`` and
            batch pointers in ``ptr``.
        :returns: Mean Sinkhorn loss, computed using float32 coordinates.
        :rtype: torch.Tensor
        """
        return self.loss(set1.data.to(torch.float32), set2.data.to(torch.float32), ptr_x=set1.ptr, ptr_y=set2.ptr).mean()

def minimum_ball_error(c, predicted_c, r, predicted_r, p=2, **kwargs):
    """Compute center and radius error for minimum enclosing balls.

    :param c: Ground-truth centers with shape ``(batch_size, dimension)``.
    :type c: torch.Tensor
    :param predicted_c: Predicted centers, with the same shape as ``c``.
    :type predicted_c: torch.Tensor
    :param r: Ground-truth radii.
    :type r: torch.Tensor
    :param predicted_r: Predicted radii, with the same shape as ``r``.
    :type predicted_r: torch.Tensor
    :param p: Unused. Center error always uses the Euclidean norm.
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Mean Euclidean center error plus mean squared radius error.
    :rtype: torch.Tensor
    """
    err = torch.mean(torch.linalg.norm(c - predicted_c, axis=1)) + mse_loss(r, predicted_r)
    return err 

def minimum_ball_radius_error(r, predicted_r, **kwargs):
    """Compute mean squared radius error for minimum enclosing balls.

    :param r: Ground-truth radii.
    :type r: torch.Tensor
    :param predicted_r: Predicted radii, with the same shape as ``r``.
    :type predicted_r: torch.Tensor
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Mean squared error between predicted and ground-truth radii.
    :rtype: torch.Tensor
    """
    err = mse_loss(r, predicted_r)
    return err

def minimum_annulus_error(c, predicted_c, inner_r, predicted_inner_r, outer_r, predicted_outer_r, **kwargs):
    """Compute center and radius error for minimum enclosing annuli.

    :param c: Ground-truth centers with shape ``(batch_size, dimension)``.
    :type c: torch.Tensor
    :param predicted_c: Predicted centers, with the same shape as ``c``.
    :type predicted_c: torch.Tensor
    :param inner_r: Ground-truth inner radii.
    :type inner_r: torch.Tensor
    :param predicted_inner_r: Predicted inner radii.
    :type predicted_inner_r: torch.Tensor
    :param outer_r: Ground-truth outer radii.
    :type outer_r: torch.Tensor
    :param predicted_outer_r: Predicted outer radii.
    :type predicted_outer_r: torch.Tensor
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Mean Euclidean center error plus the mean squared errors
        for the inner and outer radii.
    :rtype: torch.Tensor
    """
    err = torch.mean(torch.linalg.norm(c - predicted_c, axis=1)) + minimum_annulus_radius_error(inner_r, predicted_inner_r, outer_r, predicted_outer_r)
    return err

def minimum_annulus_radius_error(inner_r, predicted_inner_r, outer_r, predicted_outer_r, **kwargs):
    """Compute radius error for minimum enclosing annuli.

    :param inner_r: Ground-truth inner radii.
    :type inner_r: torch.Tensor
    :param predicted_inner_r: Predicted inner radii.
    :type predicted_inner_r: torch.Tensor
    :param outer_r: Ground-truth outer radii.
    :type outer_r: torch.Tensor
    :param predicted_outer_r: Predicted outer radii.
    :type predicted_outer_r: torch.Tensor
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Sum of the mean squared errors for the inner and outer radii.
    :rtype: torch.Tensor
    """
    inner_r_err = mse_loss(inner_r, predicted_inner_r)
    outer_r_err = mse_loss(outer_r, predicted_outer_r)
    return inner_r_err + outer_r_err

def minimum_enclosing_ellipse_error(**kwargs):
    """Placeholder for minimum enclosing ellipse error.

    :param kwargs: Additional keyword arguments; ignored.
    :raises NotImplementedError: Always, because this loss is not implemented.
    """
    raise NotImplementedError


def scatter_binary_cross_entropy(predicted, target, index, reduction="mean", **kwargs):
    """Compute binary cross-entropy with logits, reduced by set.

    :param predicted: Predicted logits.
    :type predicted: torch.Tensor
    :param target: Binary targets, with the same shape as ``predicted``.
    :type target: torch.Tensor
    :param index: Set indices used to group elementwise losses.
    :type index: torch.Tensor
    :param reduction: Reduction passed to ``torch_scatter.scatter``.
    :type reduction: str
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Mean of the grouped binary cross-entropy losses.
    :rtype: torch.Tensor
    """
    unrolled = binary_cross_entropy_with_logits(input=predicted, target=target, reduction="none")
    per_set_bce_error=scatter(src=unrolled, index=index, reduce=reduction)
    return per_set_bce_error.mean()

def binary_focal_loss(input, target, gamma=5, reduction=None, **kwargs):
    """Compute focal loss for binary classification from logits.

    :param input: Predicted logits.
    :type input: torch.Tensor
    :param target: Binary targets, with the same shape as ``input``.
    :type target: torch.Tensor
    :param gamma: Focusing exponent applied to ``1 - p_t``, where ``p_t``
        is the probability assigned to the target class. Defaults to 5.
    :type gamma: float
    :param reduction: ``None`` returns elementwise losses, ``'sum'`` returns
        their sum, and any other value returns their mean.
    :type reduction: str or None
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Elementwise or reduced focal loss, depending on ``reduction``.
    :rtype: torch.Tensor
    """
    probs = torch.sigmoid(input)
    targets = target.float()

    # Compute binary cross entropy
    bce_loss = binary_cross_entropy_with_logits(input, targets, reduction='none')

    # Compute focal weight
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    loss = focal_weight * bce_loss
    if reduction is None:
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()

def scatter_binary_focal_loss(predicted, target, index, reduction='mean', **kwargs):
    """Compute binary focal loss with exponent 5, reduced by set.

    :param predicted: Predicted logits.
    :type predicted: torch.Tensor
    :param target: Binary targets, with the same shape as ``predicted``.
    :type target: torch.Tensor
    :param index: Set indices used to group elementwise losses.
    :type index: torch.Tensor
    :param reduction: Reduction passed to ``torch_scatter.scatter``.
    :type reduction: str
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Mean of the grouped focal losses.
    :rtype: torch.Tensor
    """
    unrolled = binary_focal_loss(input=predicted, target=target)
    per_set_fl_error = scatter(src=unrolled, index=index, reduce=reduction)
    return per_set_fl_error.mean()

def directional_width_loss(predicted, target, in_dim,n=100, **kwargs):
    """Measure differences in projection extrema along random directions.

    Directions are sampled uniformly from the unit sphere. For each set,
    the absolute differences in minimum and maximum projections are summed
    over all sampled directions, then averaged over the batch.

    :param predicted: Predicted point-cloud batch, with coordinates in
        ``data`` and set indices in ``batch1``.
    :param target: Ground-truth point-cloud batch, with coordinates in
        ``data`` and set indices in ``batch1``.
    :param in_dim: Dimension of the point coordinates.
    :type in_dim: int
    :param n: Number of sampled directions.
    :type n: int
    :param kwargs: Additional keyword arguments; ignored.
    :returns: Mean summed error in projection extrema across the batch.
    :rtype: torch.Tensor

    .. seealso::

        `Coreset survey
        <https://users.cs.duke.edu/~pankaj/publications/surveys/coreset-survey.pdf>`_
        for background on directional width.
    """
    uniform_sphere_dist = uniform_direction(dim=in_dim)
    directions = torch.tensor(uniform_sphere_dist.rvs(n), dtype=torch.float32, device=target.data.device)
    
    proj_predicted = torch.matmul(predicted.data, directions.T)
    proj_target = torch.matmul(target.data, directions.T)


    max_p = scatter(src= proj_predicted, index = predicted.batch1, reduce='max', dim=0) #add batch indicator manually
    max_q = scatter(src= proj_target, index = target.batch1, reduce='max', dim=0)

    min_p = -1 * scatter(src = -1 * proj_predicted, index = predicted.batch1, reduce='max', dim=0)
    min_q = -1 * scatter(src = -1 * proj_target, index = target.batch1, reduce='max', dim=0)

    diff_max = torch.abs(max_q - max_p)
    diff_min = torch.abs(min_q - min_p)

    losses = torch.sum(diff_max + diff_min, dim=1)
    
    return torch.mean(losses)
