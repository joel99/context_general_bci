import os
from pathlib import Path

import logging # we use top level logging since most actual diagnostic info is in libs
import hydra
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from pytorch_lightning.loggers import WandbLogger
import wandb

from config import RootConfig
from data import SpikingDataset
from model import BrainBertInterface
from utils import get_latest_ckpt_from_wandb_id

@hydra.main(version_base=None, config_path='config', config_name="config")
def run_exp(cfg : RootConfig) -> None:
    logging.info(f"Running Gelsight, dumping config:")
    logging.info(OmegaConf.to_yaml(cfg))
    pl.seed_everything(seed=cfg.seed)

    dataset = SpikingDataset(cfg.dataset)
    train, val = dataset.create_tv_datasets()
    logging.info(f"Training on {len(train)} examples")

    model = BrainBertInterface(cfg.model, dataset.get_data_attrs())

    epochs = cfg.train.epochs
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            filename='val-{epoch:02d}-{val_loss:.4f}',
            save_top_k=2,
            mode='min',
            every_n_epochs=10,
            dirpath=None
        )
    ]

    if cfg.train.patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=cfg.train.patience, # Learning can be fairly slow, larger patience should allow overfitting to begin (which is when we want to stop)
                min_delta=0.00005, # we can tune this lower to squeeze a bit more..
            )
        )

    logger = WandbLogger(project=cfg.wandb_project)

    pl.seed_everything(seed=cfg.seed)

    if cfg.train.steps:
        max_steps = cfg.train.steps
        epochs = None
    else:
        max_steps = -1

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        val_check_interval=1.0,
        callbacks=callbacks,
        default_root_dir=cfg.default_root_dir,
        track_grad_norm=2 if cfg.train.log_grad else -1,
        precision=16 if cfg.model.half_precision else 32,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_batches
    )

    if torch.cuda.device_count() <= 1 or trainer.global_rank == 0:
        # Note, wandb.run can also be accessed as logger.experiment but there's no benefit
        if cfg.tag:
            wandb.run.name = f'{cfg.tag}-{wandb.run.id}'
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # === Train ===
    num_workers = 0
    # num_workers = len(os.sched_getaffinity(0)) # If this is set too high, the dataloader may crash.
    logging.info("Preparing to fit...")
    trainer.fit(
        model,
        DataLoader(
            train, shuffle=True,
            batch_size=cfg.train.batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=train.collater_factory()
        ),
        DataLoader(val,
            batch_size=cfg.train.batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            collate_fn=val.collater_factory()
        ),
        ckpt_path=get_latest_ckpt_from_wandb_id(cfg.wandb_project, cfg.load_from_id) if cfg.load_from_id else None
    )
    logging.info('Run complete')

if __name__ == '__main__':
    run_exp()