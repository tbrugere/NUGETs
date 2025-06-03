from typing import Set
from torch.utils.data import DataLoader

import lightning as pl

from nugets.models import BackBone, Model
from nugets.models.backbones.transformer import Transformer
from nugets.models.backbones.sumformer import Sumformer
from nugets.tasks.dummy_tasks import SetIdentityTask

def test_transformer():

    assert issubclass(Model, pl.LightningModule)

    task = SetIdentityTask(
            "GrowingCircles", 
            dict(
                dim=2,
                min_points=10,
                max_points=20,
                radius="linear", 
                length= 64, 
                )
            )
    
    backbone: BackBone = Transformer(
            n_heads=4,
            n_layers=2,
            d_model=64,
        #     key_dim=16,
            feed_forward_hidden_dim=256, 
            )

    model = Model(backbone, task, batch_size=32, learning_rate=1e-3, debug_mode=True)
    additional_options = dict()
    trainer = pl.Trainer(default_root_dir="workdir/", 
                         max_epochs=1, 
                         limit_train_batches=2,
                         log_every_n_steps=1,
                         gradient_clip_val=0.01, 
                         precision="16-mixed",
                         logger=False, 
                         **additional_options)

    trainer.fit(model=model)

def test_sumformer():
    assert issubclass(Model, pl.LightningModule)

    task = SetIdentityTask(
            "GrowingCircles", 
            dict(
                dim=2,
                min_points=10,
                max_points=20,
                radius="linear", 
                length= 64, 
                )
            )

    backbone: BackBone = Sumformer(
        n_layers = 2,
        d_model=64,
        feed_forward_hidden_dim=256
    )
    
