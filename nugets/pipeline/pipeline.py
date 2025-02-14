
def train_model(model):
    import lightning as pl
    trainer = pl.Trainer(default_root_dir="workdir/")
    trainer.fit(model=model)
