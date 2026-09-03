from typing import Literal
from pathlib import Path

from torch_geometric.datasets import ShapeNet as Pyg_ShapeNet
from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from .dataset_utils import download_from_url, extract_data_from_zip, extract_nested_zips
from nugets.datasets.datapoint_types import Set_datapoint
from nugets.datasets.register import register as dataset_register
import numpy as np
import torch

# TODO: Raw RNAseq dataset does not download to correct place. Should be workdir/datasets/raw/

@dataset_register
class RNASeqPointCloud(Dataset[Set_datapoint]):
    """
    High dimensional RNAseq data. 
    """
    datatype: Set_datapoint
    seed: int = 42 # random seed set for sampling point clouds. 
    length: int | None = None
    split_seed: int = 42
    default_root = 'workdir/datasets'

    _HF_REPO_ID = "geometricdataset/Neural-CG-Benchmark"
    _HF_PREFIX = "server-local"

    def _auto_download(self, dest: str | Path | None = None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for automatic dataset downloads."
            ) from exc

        if dest is not None:
            download_dest = Path(dest)
        else:
            parts = self.root.parts
            if self._HF_PREFIX in parts:
                idx = parts.index(self._HF_PREFIX)
                download_dest = Path(".") if idx == 0 else Path(*parts[:idx])
            else:
                download_dest = Path("data")

        snapshot_download(
            repo_id=self._HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{self._HF_PREFIX}/rna.npy"],
            local_dir=str(download_dest),
            local_dir_use_symlinks=False,
        )
        
    
    def __init__(self, length=100, size=100, which="train", seed=42, 
                 auto_download:bool = True, **kwargs):
        
        self.root = Path(f'{self.default_root}/raw/rna.npy')
        if not self.root.exists() and auto_download:
            self._auto_download(self.root)
        elif not self.root.exists() and not auto_download:
            raise FileNotFoundError(f'RNAseq data not found in {self.root}. Please download from https://huggingface.co/datasets/geometricdatasets/Neural-CG-Benchmark')

        raw_data = np.load(self.root)

        rng = np.random.default_rng(seed=42)
        self.length = length
        self.size = size

        selected_indices = rng.choice(len(raw_data), size = (self.length, self.size))
        self.inner = raw_data[selected_indices]
        self.size = size
        self.dimension = raw_data.shape[1]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        dp = self.inner[i]
        return Set_datapoint(pointset=dp)

    def dataset_parameters(self):
        return {'dim': self.dimension, 'size': self.size, 'length':self.length}