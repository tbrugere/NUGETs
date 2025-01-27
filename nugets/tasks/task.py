from typing import Literal
from ml_lib.datasets import Dataset

from nugets.datasets import get_dataset_register

class Task():

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
        
    def get_inner_dataset(self, which: Literal["train", "val", "test"]) -> Dataset:
        """Get the inner dataset"""
        dataset_register = get_dataset_register()
        dataset_type = dataset_register[self.dataset_name]
        return dataset_type(**self.dataset_parameters, which=which)

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Any preprocessing of the dataset that can be cached should be applied here"""
        return dataset
        
    def get_cached_processed_dataset_path(self, which: Literal["train", "val", "test"]) -> str:
        """Get the path to the cached processed dataset"""
        task_name = self.__class__.__name__
        task_hash = hash(self)
        return f"workdir/datasets/processed/{task_name}_{task_hash}_{which}.tar"

    def cache_processed_dataset(self, which: Literal["train", "val", "test"]):
        """Cache the processed dataset"""
        from ml_lib.datasets.datasets.tar_dataset import AutoTarDataset
        dataset = self.get_inner_dataset(which)
        processed_dataset = self.process_dataset(dataset)
        path = self.get_cached_processed_dataset_path(which)
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
        

    """
    Encoding
    --------
    """

    def get_encoder_decoder(self, backbone):
        """Get the encoder-decoder for the task"""
        del backbone
        raise NotImplementedError
    



