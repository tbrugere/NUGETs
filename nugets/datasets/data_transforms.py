from typing import Optional
import functools as ft
import numpy as np
from numpy.random import default_rng; rng = default_rng()
from ml_lib.datasets.transforms import Transform, transform_register, Element, FunctionTransform

def split_indices(num_samples, *percents, rng=rng):
    if not isinstance(num_samples, int):
        num_samples = len(num_samples)
    assert np.isclose(sum(percents) , 1)
    cdf = np.cumsum(percents)
    cdf = num_samples * cdf
    shuffle = np.arange(num_samples)
    rng.shuffle(shuffle)
    indices = (shuffle[:, None] >= cdf[None, :])
    indices = indices.sum(-1)
    return indices
    
def split_arrays(split_indices, *arrays):
    all_splits = np.unique(split_indices)
    
    ret = []
    for i in all_splits:
        mask = split_indices == i
        if len(arrays) == 1:
            array, = arrays
            ret.append(array[mask])
        else:
            ret.append([array[mask] for array in arrays])
    if len(ret) == 1:
        ret, = ret
    return ret

@transform_register
class SplitTransform(Transform[Element, Element]):
    """should always spit the same split if given the same seed / percents and dataset_size"""
    splits: list[str]
    percents: list[float]
    which:str
    seed: int 
    
    map : np.ndarray|None
    indices: np.ndarray|None

    def __init__(self, which, seed=42, splits: list[str]=["train", "eval", "test"], percents=[.9, .05, .05]):
        super().__init__()
        if which not in splits:
            raise ValueError(f"{which} not in the possible splits {splits}")
        assert len(percents) == len(splits)
        self.splits = splits
        self.which = which
        self.percents = percents
        self.seed = seed
        self.map = None
        self.indices = None

    def _initialize(self):
        inner = self.inner
        n = len(inner)
        rng = default_rng(self.seed)
        map = self.map = split_indices(n, *self.percents, rng=rng)
        split_number = self.splits.index(self.which)
        self.indices,  = np.nonzero(map == split_number)

    def __len__(self):
        assert self.indices is not None
        return len(self.indices)

    def __getitem__(self, i):
        inner = self.inner
        indices = self.indices
        assert indices is not None
        return inner[indices[i]]
    

