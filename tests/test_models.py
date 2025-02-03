from typing import Set
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from nugets.models import BackBone, Model
from nugets.models.backbones.transformer import Transformer
from nugets.tasks.dummy_tasks import SetIdentityTask

def test_transformer():
    task = SetIdentityTask(
            "GrowingCircles", 
            dict(
                dim=2,
                min_points=10,
                max_points=20,
                radius="linear"
                )
            )
    
    backbone: BackBone = Transformer(
            n_heads=4,
            n_layers=2,
            d_model=64,
            key_dim=16,
            feed_forward_hidden_dim=256
            )

    model = Model(backbone, task, batch_size=32, learning_rate=1e-3)

    trainer = pl.Trainer(max_epochs=1, limit_train_batches=2)
    trainer.fit(model=model)


