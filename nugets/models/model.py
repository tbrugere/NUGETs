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

from ml_lib.datasets.datapoint import Datapoint
import lightning as pl
import torch
from torch import nn

if TYPE_CHECKING:
    from nugets.tasks import Task

from .backbone import BackBone


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

    def __init__(self, backbone: BackBone, task: "Task"):
        super().__init__()
        self.backbone = backbone
        self.encoder_decoder = task.get_encoder_decoder(backbone)

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
    
    def 
