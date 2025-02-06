from typing import Literal
import warnings
from logging import getLogger
from pathlib import Path

from ml_lib.datasets import Dataset

from nugets.datasets import get_dataset_register

log = getLogger(__name__)


class Task():
    """Problem instance"""

    dataset_name: str
    dataset_parameters: dict

    def __init__(self, dataset, dataset_parameters):
        self.dataset_name = dataset
        self.dataset_parameters = dataset_parameters

    def __hash__(self):
        """Hash the task"""
        name_hash = hash(self.dataset_name)
        parameters_hash = hash(frozenset(self.dataset_parameters.items()))
        return hash((name_hash, parameters_hash))

    """
    Dataset processing
    ------------------
    """

    _inner_datasets = {}
    _processed_datasets = {}
        
    def get_inner_dataset(self, which: Literal["train", "val", "test"]) -> Dataset:
        """Get the inner dataset"""
        if which in self._inner_datasets:
            return self._inner_datasets[which]
        dataset_register = get_dataset_register()
        dataset_type = dataset_register[self.dataset_name]
        dataset = dataset_type(**self.dataset_parameters, which=which)
        self._inner_datasets[which] = dataset
        return dataset

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Any preprocessing of the dataset that can be cached should be applied here"""
        return dataset

    def get_cached_processed_dataset_path(self, which: Literal["train", "val", "test"]) -> Path:
        """Get the path to the cached processed dataset"""
        task_name = self.__class__.__name__
        task_hash = hash(self)
        return Path(f"workdir/datasets/processed/{task_name}_{task_hash}_{which}.tar")

    def cache_processed_dataset(self, which: Literal["train", "val", "test"], 
                                skip_if_exists=False):
        """Cache the processed dataset"""
        from ml_lib.datasets.datasets.tar_dataset import AutoTarDataset
        log.info(f"{self}: Caching processed dataset for {which}")
        path = self.get_cached_processed_dataset_path(which)
        if path.exists() and skip_if_exists:
            return
        dataset = self.get_inner_dataset(which)
        processed_dataset = self.process_dataset(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        AutoTarDataset.save_dataset(path, processed_dataset)

    def datapoint_type(self):
        """Get the type of the datapoint"""
        dataset_register = get_dataset_register()
        dataset_type = dataset_register[self.dataset_name]
        return dataset_type.datatype

    def get_cached_processed_dataset(self, which: Literal["train", "val", "test"]) -> Dataset:
        """Return the cached processed dataset"""
        from ml_lib.datasets.datasets.tar_dataset import AutoTarDataset
        path = self.get_cached_processed_dataset_path(which)
        return AutoTarDataset(self.datapoint_type(), path)

    def get_any_cached_processed_dataset(self):
        for which in ("train", "val", "test"):
            path = self.get_cached_processed_dataset_path(which)
            if path.exists():
                return self.get_cached_processed_dataset(which)
        return None

    def get_dataset(self, which: Literal["train", "val", "test"]) -> Dataset:
        """Get the dataset"""
        if which in self._processed_datasets:
            return self._processed_datasets[which]
        cached_dataset_path = self.get_cached_processed_dataset_path(which)
        if not cached_dataset_path.exists():
            self.cache_processed_dataset(which)

        self._processed_datasets[which] = self.get_cached_processed_dataset(which)
        return self._processed_datasets[which]
        
    def dataset_info(self):
        """Get the dataset info"""
        if len(self._processed_datasets) != 0:
             ds = list(self._processed_datasets.values())[0]  
        elif len(self._inner_datasets) != 0:
            ds = list(self._inner_datasets.values())[0]
        elif (ds:= self.get_any_cached_processed_dataset()) is not None: 
            pass
        else: 
            warnings.warn("Loading train dataset to get dataset info")
            ds = self.get_inner_dataset("train")
        return ds.dataset_parameters()

    def prepare_data(self):
        for which in "train", "val", "test":
            self.cache_processed_dataset(which, skip_if_exists=True)
            


    """
    Data loading
    ------------
    """

    def get_dataloader(self, which: Literal["train", "val", "test"], batch_size: int, 
                       no_workers=False):
        """Get the dataloader"""
        from torch.utils.data import DataLoader
        dataset = self.get_dataset(which)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4 if not no_workers else 0, 
                          pin_memory=not no_workers, 
                          collate_fn=dataset.collate)
           

    """
    Encoding
    --------
    """

    def get_encoder_decoder(self, backbone):
        """Get the encoder-decoder for the task"""
        del backbone
        raise NotImplementedError
    



