from nugets.pipeline.configs import Config

def train_model(model):
    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger
    global_config = Config.get()
    wandb_logger = WandbLogger(project=global_config.wandb_project)
    trainer = pl.Trainer(default_root_dir="workdir/", logger=wandb_logger)
    trainer.fit(model=model)
