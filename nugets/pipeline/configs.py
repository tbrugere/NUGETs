import hashlib
from typing import TYPE_CHECKING, Any
from logging import getLogger
from pathlib import Path
import pydantic
import yaml

from ml_lib.misc.data_structures import SingletonMeta

if TYPE_CHECKING:
    from google.cloud.storage import Client as GCS_Client, Bucket as GCS_Bucket

log = getLogger(__name__)

class GlobalConf(pydantic.BaseModel):
    """Object representing the global config for NUGETs.

    This should generally be loaded using the :py:class:`Config` class below, 
    specifically the :py:func:`Config.load` method is run once 
    at the beginning of the program, 
    and the  :py:func:`Config.get` method subsequently 
    returns config contents in a ``GlobalConf`` object.
    """
    

    loglevel: str = "WARNING"
    wandb_key: str|None = pydantic.Field(default=None)
    wandb_project: str|None = "test-wandb"
    # gcp_key: str|None = pydantic.Field(default=None) 
    # gcp should be initialized using the 
    # gcloud auth activate-service-account 
    # command if on servers, or the gcloud auth login command when working locally
    model_config = pydantic.ConfigDict(extra="forbid")
    checkpoint_bucket: str|None = None
    processed_dataset_bucket: str
    num_workers: int = 8
    multi_epoch_data_loader: bool = True
    prefetch_factor: int = 4

    def get_default_root_dir(self, model):
        if self.checkpoint_bucket is None:
            log.warn("No checkpoint bucket provided, defaulting to local saves")
            return f"workdir/{model.get_dirname()}"
        else:
            return f"gcs://{self.checkpoint_bucket}/{model.get_dirname()}"


class ConfigConsistentHashMixin():
    def consistent_hash(self):
        return hashlib.sha256(f"{self.__class__.__qualname__}::{self.model_dump_json()}".encode('utf-8', errors='ignore')).digest()#type:ignore
        

class TaskConf(pydantic.BaseModel, ConfigConsistentHashMixin):
    type: str
    dataset: str
    dataset_config: dict
    ood_dataset: str
    ood_config: dict
    model_config = pydantic.ConfigDict(extra="forbid")

    def load(self):
        from nugets.tasks import get_tasks_register
        register = get_tasks_register()
        task_type = register[self.type]
        return task_type(self.dataset, self.dataset_config, ood_dataset_name=self.ood_dataset, ood_dataset_parameters=self.ood_config)


class BackboneConf(pydantic.BaseModel, ConfigConsistentHashMixin):
    type: str
    model_config = pydantic.ConfigDict(extra="allow")

    def get_type(self):
        from nugets.models.backbones import get_backbones_register
        register = get_backbones_register()
        backbone_type = register[self.type]
        return backbone_type

    def get_parameters(self) -> dict[str, Any]:
        assert self.model_extra is not None
        return self.model_extra

    def load(self):
        backbone_type = self.get_type()
        return backbone_type.from_dict(self.get_parameters())

class ModelConf(pydantic.BaseModel, ConfigConsistentHashMixin):
    task: TaskConf
    backbone: BackboneConf
    batch_size: int
    learning_rate: float
    debug_mode: bool = False
    loss_function: str = 'mse_loss'
    model_config = pydantic.ConfigDict(extra="forbid")


class Config(metaclass=SingletonMeta):

    config: None| GlobalConf

    gcs_client: "None | GCS_Client"

    def __init__(self):
        self.config = None
        self.gcs_client = None

    @classmethod
    def load(cls, path:Path):
        self = cls()
        if not path.exists():
            log.warn(f"Config path {path} does not exists, using config defaults")
            self.config = GlobalConf()
            return
        with path.open() as f:
            config_dict = yaml.safe_load(f)
        self.config = GlobalConf.model_validate(config_dict)

    @classmethod
    def get(cls) -> GlobalConf:
        self = cls()
        if self.config is None:
            raise ValueError("Global config was not loaded properly")
        return self.config

    @classmethod
    def get_gcs_client(cls) -> "GCS_Client":
        from google.cloud.storage import Client
        self = cls()
        if self.gcs_client is None:
            self.gcs_client = Client()
        return self.gcs_client

    @classmethod
    def get_processed_dataset_bucket(cls) -> "GCS_Bucket":
        client = cls.get_gcs_client()
        bucket = client.bucket(cls.get().processed_dataset_bucket)
        return bucket

    @classmethod
    def get_checkpoint_bucket(cls) -> "GCS_Bucket":
        client = cls.get_gcs_client()
        bucket = client.bucket(cls.get().checkpoint_bucket)
        return bucket

