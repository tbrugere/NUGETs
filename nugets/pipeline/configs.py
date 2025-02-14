from typing import Any
from logging import getLogger
from pathlib import Path
import pydantic
import yaml

from ml_lib.misc.data_structures import SingletonMeta

log = getLogger(__name__)

class GlobalConf(pydantic.BaseModel):
    loglevel: str = "WARNING"
    wandb_key: str|None = pydantic.Field(default=None)
    wandb_project: str|None = "test-wandb"
    gcp_key: str|None = pydantic.Field(default=None)
    model_config = pydantic.ConfigDict(extra="forbid")

class TaskConf(pydantic.BaseModel):
    type: str
    dataset: str
    dataset_config: dict
    model_config = pydantic.ConfigDict(extra="forbid")

    def load(self):
        from nugets.tasks import get_tasks_register
        register = get_tasks_register()
        task_type = register[self.type]
        return task_type(self.dataset, self.dataset_config)


class BackboneConf(pydantic.BaseModel):
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

class ModelConf(pydantic.BaseModel):
    task: TaskConf
    backbone: BackboneConf
    batch_size: int
    learning_rate: float
    debug_mode: bool = False
    model_config = pydantic.ConfigDict(extra="forbid")


class Config(metaclass=SingletonMeta):

    config: None| GlobalConf

    def __init__(self):
        self.config = None

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

