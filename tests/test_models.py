from typing import Set
from torch.utils.data import DataLoader
from torch import float32

import lightning as pl

from nugets.models import BackBone, Model
from nugets.models.backbones.transformer import Transformer
from nugets.models.backbones.sumformer import Sumformer
from nugets.tasks.dummy_tasks import SetIdentityTask, SingleLabelDummyTask
from nugets.tasks.distance_tasks import WassersteinDistanceTask
from nugets.pipeline.configs import Config

from pathlib import Path

# TODO: Issue with num-workers, could have to do with debugger mode? 

def test_transformer_with_labeled_set_task():
    """
    Tests training on labelled tasks as well as aggregation step from the transformer BackBone
    TODO: throws a runtime error (device-side assert) when 'max' aggregation is used.  
    """
    assert issubclass(Model, pl.LightningModule)

    path = Path('/home/sam/NUGETs/config.yaml')
    Config.load(path)
    global_config = Config.get()

    task = SingleLabelDummyTask(
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
        #   key_dim=16,
            feed_forward_hidden_dim=256,
            aggregation = 'mean' 
            )
    model = Model(backbone, task, batch_size=32, learning_rate=1e-3, debug_mode=True)
    model = model.to(dtype=float32)
    additional_options = dict()
    trainer = pl.Trainer(default_root_dir="workdir/", 
                         max_epochs=2, 
                         limit_train_batches=2,
                         log_every_n_steps=1,
                         gradient_clip_val=0, 
                         precision="16-mixed",
                         logger=False, 
                         use_distributed_sampler=False,
                         devices=1,
                         **additional_options)

    trainer.fit(model=model)

def test_sumformer_with_labeled_set_task():
    """
    tests training on labelled tasks with SumFormer backbone
    """
    assert issubclass(Model, pl.LightningModule)

    path = Path('/home/sam/NUGETs/config.yaml')
    Config.load(path)
    global_config = Config.get()
    task = SingleLabelDummyTask(
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
        feed_forward_hidden_dim=256,
        aggregation='sum'
    )
    model = Model(backbone, task, batch_size=32, learning_rate=1e-3, debug_mode=True)
    model = model.to(dtype=float32)
    additional_options = dict()
    trainer = pl.Trainer(default_root_dir="workdir/", 
                         max_epochs=1, 
                         limit_train_batches=2,
                         log_every_n_steps=1,
                         gradient_clip_val=0, 
                         precision="16-mixed",
                         logger=False, 
                         use_distributed_sampler=False,
                         devices=1,
                         **additional_options)

    trainer.fit(model=model)
    

def test_transformer_with_dummy_set_to_set_task():

    assert issubclass(Model, pl.LightningModule)
    path = Path('/home/sam/NUGETs/config.yaml')
    Config.load(path)
    global_config = Config.get()

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
            aggregation = 'none' 
            )

    model = Model(backbone, task, batch_size=32, learning_rate=1e-3, debug_mode=True)
    additional_options = dict()
    model = model.to(dtype=float32)
    trainer = pl.Trainer(default_root_dir="workdir/", 
                         max_epochs=1, 
                         limit_train_batches=2,
                         log_every_n_steps=1,
                         gradient_clip_val=0.01, 
                         precision="16-mixed",
                         logger=False, 
                         use_distributed_sampler=False,
                         devices=1,
                         **additional_options)

    trainer.fit(model=model)

def test_sumformer_with_dummy_set_to_set_task():
    assert issubclass(Model, pl.LightningModule)
    path = Path('/home/sam/NUGETs/config.yaml')
    Config.load(path)
    global_config = Config.get()
    
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
        feed_forward_hidden_dim=256,
        aggregation='none'
    )

    model = Model(backbone, task, batch_size=32, learning_rate=1e-3, debug_mode=True)
    additional_options = dict()
    trainer = pl.Trainer(default_root_dir="workdir/", 
                         max_epochs=1, 
                         limit_train_batches=2,
                         log_every_n_steps=1,
                         gradient_clip_val=0, 
                        #  precision="16-mixed",
                         logger=False, 
                         use_distributed_sampler=False,
                         devices=1,
                         **additional_options)

    trainer.fit(model=model)

def test_siamese_nn():
    path = Path('/home/sam/NUGETs/config.yaml')
    Config.load(path)
    global_config = Config.get()

    encoder: BackBone =  Transformer(
            n_heads=4,
            n_layers=2,
            d_model=64,
        #     key_dim=16,
            feed_forward_hidden_dim=256,
            aggregation = 'none' 
            )
    
    

# TODO: Ashley+Mizuho, write a unit test for the set NN using the SingleLabelDummyTask
def test_set_nn():
    pass