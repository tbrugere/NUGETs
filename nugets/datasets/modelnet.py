from typing import Literal
from pathlib import Path
from torch import Tensor

from torch_geometric.datasets import ModelNet as Pyg_ModelNet
from torch_geometric.transforms import SamplePoints, BaseTransform
from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from nugets.datasets.datapoint_types import Graph_datapoint, Set_datapoint
from nugets.datasets.register import register


@register
class ModelNet(Dataset[Graph_datapoint]):

    datatype = Graph_datapoint

    inner: Pyg_ModelNet|SplitTransform

    def __init__(self, name: Literal["10", "40"]= "10", split_seed = 42,
                 which="train"):
        root_dir = Path("workdir/datasets/raw")
        root_dir.mkdir(exist_ok=True, parents=True)
        is_train_or_val = which in ("train", "val")
        inner: Pyg_ModelNet|SplitTransform = Pyg_ModelNet(root=str(root_dir), 
                             train=is_train_or_val, 
                             name= name)
        if is_train_or_val:
            split_transform: SplitTransform = SplitTransform(
                    which=which, seed=split_seed, 
                splits=["train", "val"], percents=[.9, .1])
            inner = split_transform(inner)

        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        dp = self.inner[i]
        return Graph_datapoint(dp.x, dp.edge_index)

@register
class ModelNetPointset(Dataset[Set_datapoint]):
    datatype = Set_datapoint
    
    def __init__(self, name: Literal["10", "40"], 
                 split_seed=42, size=1024, which='train'):
        root_dir=Path('workdir/datasets/raw')
        root_dir.mkdir(exist_ok=True, parents=True)
        is_train_or_val = which in ("train", "val")
        inner: Pyg_ModelNet|SplitTransform = Pyg_ModelNet(root=str(root_dir), 
                             train=is_train_or_val, 
                             name= name, transform=SamplePoints(num=size))
        if is_train_or_val:
            split_transform: SplitTransform = SplitTransform(
                    which=which, seed=split_seed, 
                splits=["train", "val"], percents=[.9, .1])
            inner = split_transform(inner)
        self.inner = inner
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        dp = self.inner[i]
        return Set_datapoint(pointset=dp.x)