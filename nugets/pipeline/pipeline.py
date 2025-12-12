from datetime import timedelta
from nugets.pipeline.configs import Config

def train_model(model, *, profile=False, n_epochs: int):
    import lightning as pl
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.profilers import PyTorchProfiler
    from lightning.pytorch.callbacks import ModelCheckpoint

    model_dir = model.get_dir()
    print("Model directory", model_dir)

    ## cast model parameters to float32 (need this for PyTorch Lightning AMP)
    model = model.to(torch.float32)

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
    print(global_config.get_default_root_dir(model))
    dir_cfg = global_config.get_default_root_dir(model)
    dir_cfg = 'workdir'
    checkpoint_callback = ModelCheckpoint(
            dirpath=dir_cfg, 
            # monitor="val/loss", 
            # save_top_k=10,
            save_top_k=-1,
            # every_n_epochs=1, 
            # every_n_train_steps=5, 
            save_last=True, 
            train_time_interval=timedelta(minutes=10), 
            )

    # print(global_config.get_default_root_dir(model))

    #model.save_parameters_to_cloud()
    print("Default global config directory, cloud", global_config.get_default_root_dir(model))
    root_cfg = 'workdir'
    trainer = pl.Trainer(default_root_dir=root_cfg, 
                         logger=wandb_logger, 
                         gradient_clip_val=0, 
                         max_epochs=n_epochs,
                         precision="16-mixed", 
                         # detect_anomaly=True, 
                         # profiler="simple", 
                         callbacks=[checkpoint_callback], 
                         use_distributed_sampler=False,
                         **additional_options
                         )
    trainer.fit(model=model, )
    # model.
