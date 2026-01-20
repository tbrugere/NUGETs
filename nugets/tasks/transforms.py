from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Datapoint
from ml_lib.datasets.transforms import FunctionTransform
from ml_lib.datasets import Transform

from torch import Tensor, tensor
from torch_heterogeneous_batching import Batch

from .task import Task
from .register import register
from nugets.datasets.datapoint_types import LabeledSetBatch, LabeledSetDatapoint

class SetLabelTransform(Transform):
    label: Callable[Tensor, Tensor]
    def __init__(self, label):
        self.label = label
    
    def __len__(self):
        return len(self.inner)
    
    def __getitem__(self, idx):
        pointset = self.inner[idx].pointset
        label = self.label(pointset)
        return LabeledSetDatapoint(pointset=pointset, label=label)