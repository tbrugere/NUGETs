from typing import Literal
from base64 import b64encode
import functools as ft
import hashlib
from io import BytesIO
from logging import getLogger
from pathlib import Path
from tqdm.auto import tqdm
import warnings

from ml_lib.datasets import Dataset
from ml_lib.datasets.utils import MultiEpochDataLoader

from nugets.datasets import get_dataset_register
from nugets.misc import dict_to_bytes
from nugets.pipeline.configs import Config, TaskConf

log = getLogger(__name__)


class Task():
    """Problem instance"""

    dataset_name: str
    dataset_parameters: dict

    def __init__(self, dataset, dataset_parameters):
        self.dataset_name = dataset
        self.dataset_parameters = dataset_parameters

    def consistent_hash(self) -> bytes:
        """Hash the task"""
        b = BytesIO()
        b.write(self.dataset_name.encode())
        b.write(bytes(1))
        b.write(dict_to_bytes(self.dataset_parameters))
        return hashlib.sha256(b.getvalue()).digest()

    def get_config(self) -> TaskConf:
        return TaskConf(
            type=self.__class__.__name__,
            dataset=self.dataset_name, 
            dataset_config=self.dataset_parameters, 
        )

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

    def get_cached_processed_dataset_filename(self, which: Literal["train", "val", "test"]) -> str:
        """Get the path to the cached processed dataset"""
        task_name = self.__class__.__name__
        task_hash: bytes = self.consistent_hash()
        task_hash_b64: str = b64encode(task_hash, altchars=b':-').decode()
        return f"{task_name}_{task_hash_b64}_{which}.tar"

    def get_cached_processed_dataset_path(self, which: Literal["train", "val", "test"]) -> Path:
        """Get the path to the cached processed dataset"""
        return Path("workdir/datasets/processed") / self.get_cached_processed_dataset_filename(which)

    def cache_processed_dataset(self, which: Literal["train", "val", "test"], 
                                skip_if_exists=False, 
                                try_getting_from_cloud=False, 
                                upload_to_cloud=False):
        """Cache the processed dataset"""
        from ml_lib.datasets.datasets.tar_dataset import AutoTarDataset
        log.info(f"{self}: Caching processed dataset for {which}")
        path = self.get_cached_processed_dataset_path(which)
        if path.exists() and skip_if_exists:
            return
        if try_getting_from_cloud:
            if self.get_dataset_from_cloud(which=which, unavailable_ok=True):
                return
        dataset = self.get_inner_dataset(which)
        processed_dataset = self.process_dataset(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        pbar = ft.partial(tqdm, desc=f"caching {which} dataset", miniters=1)
        AutoTarDataset.save_dataset(path, processed_dataset, pbar=pbar)
        if upload_to_cloud:
            self.upload_dataset_to_cloud(which, overwrite=False)

    def datapoint_type(self):
        """Get the type of the datapoint"""
        dataset_register = get_dataset_register()
        dataset_type = dataset_register[self.dataset_name]
        return dataset_type.datatype

    def get_cached_processed_dataset(self, which: Literal["train", "val", "test"]) -> Dataset|None:
        """Return the cached processed dataset"""
        from ml_lib.datasets.datasets.tar_dataset import AutoTarDataset
        path = self.get_cached_processed_dataset_path(which)
        if not path.exists(): return None
        return AutoTarDataset(self.datapoint_type(), path)

    def get_any_cached_processed_dataset(self):
        for which in ("train", "val", "test"):
            ds = self.get_cached_processed_dataset(which)
            if ds is not None: return ds
        return None

    def get_dataset(self, which: Literal["train", "val", "test"]) -> Dataset:
        """Get the dataset"""
        if which in self._processed_datasets:
            return self._processed_datasets[which]
        cached_dataset_path = self.get_cached_processed_dataset_path(which)
        if not cached_dataset_path.exists():
            log.info(f"{which} dataset has not been precomputed yet, precomputing")
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
            self.cache_processed_dataset(which, skip_if_exists=True, 
                                         try_getting_from_cloud=False, 
                                         upload_to_cloud = False)

    def get_dataset_cloud_object(self, which):
        bucket = Config.get_processed_dataset_bucket()
        filename = self.get_cached_processed_dataset_filename(which)
        return  bucket.blob(filename)

    def get_dataset_from_cloud(self, which, unavailable_ok=True):
        blob = self.get_dataset_cloud_object(which)
        if not blob.exists():
            if not unavailable_ok:
                raise FileNotFoundError("did not find processed dataset on google cloud")
            return False
        file = self.get_cached_processed_dataset_path(which)
        blob.download_to_filename(str(file))
        return True
        
    def upload_dataset_to_cloud(self, which, overwrite=True):
        blob = self.get_dataset_cloud_object(which)
        if blob.exists() and not overwrite:
            return 
        file = self.get_cached_processed_dataset_path(which)
        blob.upload_from_filename(str(file))

    def get_datasets_from_cloud(self, unavailable_ok=True, skip_if_exists=True):
        for which in "train", "val", "test":
            if self.get_cached_processed_dataset_path(which).exists() and skip_if_exists:
                continue
            self.get_dataset_from_cloud(which, unavailable_ok=unavailable_ok)

    def upload_datasets_to_cloud(self, overwrite=False):
        for which in "train", "val", "test":
            self.upload_dataset_to_cloud(which, overwrite=overwrite)



    """
    Data loading
    ------------
    """

    def get_dataloader(self, which: Literal["train", "val", "test"], batch_size: int, 
                       no_workers=False):
        """Get the dataloader"""
        from torch.utils.data import DataLoader
        dataset = self.get_dataset(which)
        config = Config.get()
        if config.multi_epoch_data_loader:
            data_loader = MultiEpochDataLoader
        else: data_loader = DataLoader
        return data_loader(dataset, batch_size=batch_size, shuffle=which =="train", 
                          num_workers=config.num_workers if not no_workers else 0, 
                          pin_memory=not no_workers, 
                          persistent_workers=not no_workers, 
                          prefetch_factor= None if no_workers else config.prefetch_factor, 
                          collate_fn=dataset.collate)


    """
    Encoding
    --------
    """

    def get_encoder_decoder(self, backbone):
        """Get the encoder-decoder for the task"""
        del backbone
        raise NotImplementedError
    

    """
    Metrics
    --------
    """

    def compute_metrics(self, datapoint, results):
        del datapoint, results
        raise NotImplementedError

    


