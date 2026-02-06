from typing import Literal
from pathlib import Path

from torch_geometric.datasets import ShapeNet as Pyg_ShapeNet
from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from nugets.datasets.datapoint_types import Set_datapoint
from nugets.datasets.register import register as dataset_register


@dataset_register
class ShapeNet(Dataset[Set_datapoint]):

    datatype = Set_datapoint

    inner: Pyg_ShapeNet|SplitTransform

    def __init__(self, name: Literal["10", "40"]= "10", split_seed = 42,
                 which="train"):
        root_dir = Path("workdir/datasets/raw")
        root_dir.mkdir(exist_ok=True, parents=True)
        is_train_or_val = which in ("train", "val")
        inner: Pyg_ShapeNet|SplitTransform = Pyg_ShapeNet(root=str(root_dir), 
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
        return Set_datapoint(dp.x)

        