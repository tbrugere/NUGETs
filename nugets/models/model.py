"""Base class for all models.

In this project, a model is broken down into two parts:

    1. An encoder-decoder
    2. A backbone

The encoder and decoder are Task-specific, while the backbone is shared across tasks.

They all can contain learnable parameters, but the encoder-decoder should have

- as few parameters as possible
- no hyperparameters. Its architecture should be determined by 
    - the task
    - the backbone hyperparameters

For example the encoder
"""
from typing import Any, TYPE_CHECKING, TypeVar
from logging import getLogger
from pathlib import Path
from base64 import b64encode
import yaml

from torch.optim import AdamW
from ml_lib.datasets.datapoint import Datapoint
import lightning as pl
import torch
from torch import nn

from nugets.pipeline.configs import ModelConf, TaskConf, BackboneConf

if TYPE_CHECKING:
    from nugets.tasks import Task

from .backbone import BackBone

log = getLogger(__name__)

class EncoderDecoder(nn.Module):
    """Base class for encoder-decoder models

    Encoder-decoder models have:

        - a set of hyperparameters
        - learnable parameters

    """

    def encode(self, batch: Datapoint) -> tuple[Any, Any]:
        del batch
        raise NotImplementedError

    def decode(self, backbone_result: Any) -> Any:
        del backbone_result
        raise NotImplementedError

    def compute_loss(self, batch: Datapoint,  
                     backbone_result: Any, encoder_info: Any) -> torch.Tensor:
        del batch, backbone_result, encoder_info
        raise NotImplementedError

class EncoderDecoderWithProjection(EncoderDecoder):
    """Base class for encoder-decoder models

    Encoder-decoder models have:

        - a set of hyperparameters
        - learnable parameters

    """

    in_proj: nn.Linear
    out_proj: nn.Linear
    
    def __init__(self, input_dim: int, backbone_input_dim: int,
                 backbone_output_dim: int, output_dim: int|None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.in_proj = nn.Linear(input_dim, backbone_input_dim)
        self.out_proj = nn.Linear(backbone_output_dim, output_dim)


    def decode(self, backbone_result: Any) -> Any:
        return self.out_proj(backbone_result)

    def compute_loss(self, batch: Datapoint,  
                     backbone_result: Any, encoder_info: Any) -> torch.Tensor:
        del batch, backbone_result, encoder_info
        raise NotImplementedError

class Model(pl.LightningModule):
    """Base class for all models

    Models have:

        - a backbone
        - an encoder-decoder
        - learnable parameters

    """
    task: "Task"
    encoder_decoder: nn.Module
    backbone: BackBone
    
    ####### training parameters
    batch_size: int
    learning_rate: float
    debug_mode: bool # disables a bunch of optimizations

    def __init__(self, backbone: BackBone, task: "Task", 
                 batch_size: int, learning_rate: float, 
                 debug_mode=False):
        super().__init__()
        self.backbone = backbone
        self.encoder_decoder = task.get_encoder_decoder(backbone)
        self.task = task
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.debug_mode = debug_mode
        self.save_hyperparameters(self.get_config().model_dump())
        # print(self.hparams)

    def forward(self, batch: Datapoint) -> Any:
        """Forward pass of the model"""
        encoded, _ = self.encoder_decoder.encode(batch)
        backbone_result, _ = self.backbone(encoded)
        return self.encoder_decoder.decode(backbone_result)

    def training_step(self, batch, batch_idx):
        """Training step of the model"""
        encoded, encoder_info = self.encoder_decoder.encode(batch)
        backbone_result, reg_loss = self.backbone(encoded, return_reg_loss=True)
        loss = self.encoder_decoder.compute_loss(batch, backbone_result, encoder_info)
        self.log('train_loss', loss)
        if reg_loss is not None: 
            self.log('train_reg_loss', reg_loss)
            loss = loss + reg_loss
        return loss

    def configure_optimizers(self):
        backbone_optim = self.backbone.configure_optimizer()
        if backbone_optim is not NotImplemented:
            return backbone_optim(self.parameters())
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-7)
        return optimizer

    def prepare_data(self):
        """Prepare the data"""
        self.task.prepare_data()

    def train_dataloader(self):
        """Get the training dataloader"""
        return self.task.get_dataloader("train", self.batch_size, no_workers=self.debug_mode)

    def val_dataloader(self):
        """Get the training dataloader"""
        return self.task.get_dataloader("val", self.batch_size, no_workers=self.debug_mode)

    def test_dataloader(self):
        """Get the training dataloader"""
        return self.task.get_dataloader("test", self.batch_size, no_workers=self.debug_mode)


    @classmethod
    def argument_parser(cls, parser):
        from nugets.models.backbones import get_backbones_register
        from nugets.tasks import get_tasks_register
        backbone_register = get_backbones_register()
        task_register = get_tasks_register()
        """Adds the ability to load a Model class to a parser"""

        task_group = parser.add_argument_group(title="Task", prefix=None, dest_group="task")
        backbone_group = parser.add_argument_group(title="Backbone", prefix="backbone", dest_group="backbone")
        training_param_group = parser.add_argument_group(title="Training parameters", prefix=None, dest_group="train")

        def add_backbone_parameters(args):
            backbone_name = args.type
            backbone_type = backbone_register[backbone_name]
            backbone_type.argument_parser(backbone_group)


        task_group.add_argument("--task", type=str, required=True, help="The task to train on", choices=task_register.keys())
        task_group.add_argument("--dataset", type=Path, metavar="config_path", required=True, help="The dataset to train on")
        backbone_group.add_argument("--type", type=str, required=True, help="The backbone to use", update=add_backbone_parameters, choices=backbone_register.keys())
        training_param_group.add_argument("--batch-size", type=int, required=True, help="The batch size")
        training_param_group.add_argument("--learning-rate", type=float, required=True, help="The learning rate")
        return parser

    @classmethod
    def from_args(cls, args):
        from nugets.models.backbones import get_backbones_register
        from nugets.tasks import get_tasks_register
        backbone_register = get_backbones_register()
        task_register  = get_tasks_register()
        backbone_name = args.backbone.type
        backbone_type = backbone_register[backbone_name]
        backbone = backbone_type.from_args(args.backbone)

        with args.task.dataset.open() as f:
            dataset_config = yaml.safe_load(f)
        dataset_name = dataset_config.pop("type")
        task_type = task_register[args.task.task]
        task = task_type(dataset_name, dataset_config)
        
        return cls(backbone=backbone, task=task, batch_size=args.train.batch_size, 
                   learning_rate=args.train.learning_rate)

    @classmethod
    def from_dict(cls, config: dict):
        config_ = ModelConf.model_validate(config)
        return cls.from_config(config_)

    @classmethod
    def from_config_file(cls, config: Path):
        match config.suffix:
            case "yaml":
                import yaml
                with config.open() as f:
                    config_dict = yaml.safe_load(f)
            case "json":
                import json
                with config.open() as f:
                    config_dict = json.load(f)
            case other_extension:
                print(f"unrecognized extension {other_extension}, interpreting as json")
                import json
                with config.open() as f:
                    config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_config(cls, config: ModelConf):
        return cls(backbone=config.backbone.load(), task = config.task.load(), 
                 batch_size= config.batch_size, learning_rate= config.learning_rate, 
                 debug_mode=config.debug_mode)

    def get_config(self):
        backbone_conf = self.backbone.get_config()
        task_conf = self.task.get_config()
        return ModelConf(
            backbone=backbone_conf, 
            task=task_conf, 
            batch_size = self.batch_size, learning_rate = self.learning_rate, 
            debug_mode = self.debug_mode
        )

    def consistent_hash(self):
        config = self.get_config()
        return config.consistent_hash()

    def get_dirname(self):
        return b64encode(self.consistent_hash(), altchars=b':-').decode()

    def get_dir(self, workdir=Path("workdir")) -> Path:
        dirname = self.get_dirname()
        dir = workdir / "models" / dirname
        dir.mkdir(parents=True, exist_ok=True)
        return dir


