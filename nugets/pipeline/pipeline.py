from datetime import timedelta
from nugets.pipeline.configs import Config

def train_model(model, *, profile=False, n_epochs: int):
    import lightning as pl
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.profilers import PyTorchProfiler
    from lightning.pytorch.callbacks import ModelCheckpoint

    model_dir = model.get_dir()

    global_config = Config.get()
    wandb_logger = WandbLogger(project=global_config.wandb_project, 
                               save_dir=model_dir, 
                               )
    torch.set_float32_matmul_precision('medium')

    if profile:
        profiler = PyTorchProfiler(emit_nvtx=True)
        additional_options = dict(
            profiler=profiler, 
            max_epochs = 5, 
            limit_train_batches=10
                )
    else: 
        additional_options = dict()

    checkpoint_callback = ModelCheckpoint(
            # dirpath=f"gcs://{global_config.checkpoint_bucket}/{model.get_dirname()}",
            monitor="val/loss", 
            save_top_k=10,
            save_last=True, train_time_interval=timedelta(minutes=15), 
            )


    trainer = pl.Trainer(default_root_dir=global_config.get_default_root_dir(model), 
                         logger=wandb_logger, 
                         gradient_clip_val=.01, 
                         max_epochs=n_epochs, 
                         precision="16-mixed", 
                         # detect_anomaly=True, 
                         # profiler="simple", 
                         callbacks=[checkpoint_callback], 
                         **additional_options
                         )
    trainer.fit(model=model)
    # model.
