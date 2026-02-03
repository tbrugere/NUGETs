from typing import Literal
import numpy as np
import torch
from pathlib import Path

from ml_lib.datasets.datasets.randomly_generated_dataset import GeneratedDataset
from ml_lib.datasets import Dataset
from .data_transforms import SplitTransform

from nugets.datasets.register import register as dataset_register
from .datapoint_types import Point_datapoint, Set_datapoint


@dataset_register 
class SQUIDBoundaries(Dataset[Set_datapoint]):
    """

    2D point clouds representing marine animal boundaries.
    From "Shape Queries Using Image Databases" (SQUID): https://www.cs.auckland.ac.nz/courses/compsci708s1c/lectures/Glect-html/demoCSS.html

    """
    datatype = Set_datapoint

    def __init__(self, normalization=False, split_seed=42, which="train", **kwargs):
        self.dim = 2
        # Retrieve from data folder
        inner = np.load('nugets/datasets/data/squid-data.npy')
        # TODO: Normalize incoming raw data
        if which == "ood":
            which = "val"
        is_train_or_val = which in ("train", "val") 
        if is_train_or_val:
            split_transform: SplitTransform = SplitTransform(
                    which=which, seed=split_seed, 
                    splits=["train", "val"], percents=[.9, .1])
            inner = split_transform(inner)
        self.inner = torch.tensor(inner, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inner)
    
    def __getitem__(self, i):
        dp = self.inner[i]
        return Set_datapoint(dp)
    
    def dataset_parameters(self):
        return {'dim': self.dim}
