from typing import Literal
from pathlib import Path
from torch import Tensor
import torch
import numpy as np
from collections import defaultdict

from torch_geometric.datasets import ModelNet as Pyg_ModelNet
from torch_geometric.transforms import SamplePoints, BaseTransform
from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from nugets.datasets.datapoint_types import Graph_datapoint, Set_datapoint
from nugets.datasets.register import register


@register
class ModelNet(Dataset[Graph_datapoint]):
    """
    ModelNet dataset, this version is the same as the version in PyTorch Geometric. 
    """
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
    seed: int = 42
    
    def __init__(self, name: Literal["10", "40"], 
                 split_seed=42, size=1024, length=200, which='train'):
        root_dir=Path('workdir/datasets/raw')
        root_dir.mkdir(exist_ok=True, parents=True)
        is_train_or_val = which in ("train", "val")

        rng = np.random.default_rng(seed=42)

        samples_per_class = length//int(name)
        self.length = length 
        if which in ('val', 'ood'):
            is_train = False
        else:
            is_train = True
        inner: Pyg_ModelNet|SplitTransform = Pyg_ModelNet(root=str(root_dir), 
                             train=is_train, 
                             name=name, transform=SamplePoints(num=size))
        if is_train:
            class_to_indices = defaultdict(list)
            for i in range(len(inner)):
                y = int(inner[i].y.item())
                class_to_indices[y].append(i)
            selected_indices = []
            for y in class_to_indices:
                idxs = class_to_indices[y]
                sampled_idxs = rng.choice(idxs, size=samples_per_class, replace=False)
                selected_indices.extend(sampled_idxs)
        else:
            num_selected = length//10
            selected_indices = rng.choice(len(inner), size=num_selected, replace=False)
        
        # Normalize
        selected_pts = inner[selected_indices]
        self.inner = []
        for ptset in selected_pts:
            centered = ptset.pos - ptset.pos.mean(dim=0, keepdim=True)
            scale = torch.linalg.norm(centered, dim=1).max()
            if scale > 0:
                centered = centered / scale
            self.inner.append(centered)
        self.size=size
        
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        dp = self.inner[i]
        return Set_datapoint(pointset=dp)
    
    def dataset_parameters(self):
        return {'dim': 3, 'size': self.size, 'length': self.length}
