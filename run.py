import os
import sys
from pathlib import Path
import copy
import subprocess

from pprint import pformat
import logging # we use top level logging since most actual diagnostic info is in libs
import hydra
from omegaconf import OmegaConf
import dataclasses

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb

from config import RootConfig, Metric, hp_sweep_space
from data import SpikingDataset, SpikingDataModule
from model import BrainBertInterface, load_from_checkpoint
from analyze_utils import get_best_ckpt_from_wandb_id
from utils import generate_search

r"""
    For this script
    - if you're in a slurm interactive job, or want to launch a script, directly invoke
    ```
    python run.py args
    ```

    A note on usage:
    hydra will require config_path='config' and config_name='config' to load default.

    They point to using overrides to merge in experimental config.
    `python run.py +exp=test`
    where +exp directs hydra to look for "test.yaml" in ./config/exp/

    However we'd like some additional organization
    To access subdirectory experiments, the command would look like
    `python run.py +exp/subdir=test`
    (As opposed to +exp=subdir/test)
"""
reset_early_stop = True # todo move into config

@hydra.main(version_base=None, config_path='config', config_name="config")
def run_exp(cfg : RootConfig) -> None:
    # Check for sweeping. Note we process data above because I never intend to sweep over data config.
    if cfg.sweep_cfg and os.environ.get('SLURM_JOB_ID') is None: # do not allow recursive launch
        sweep_cfg = hp_sweep_space.sweep_space[cfg.sweep_cfg]
        for cfg_trial in generate_search(sweep_cfg, cfg.sweep_trials):
            init_call = sys.argv
            init_args = init_call[init_call.index('run.py')+1:]
            additional_cli_flags = [f'{k}={v}' for k, v in cfg_trial.items()]
            meta_flags = [
                'sweep_cfg=""',
                f'sweep_tag={cfg.sweep_cfg}',
                f'tag={cfg.tag}-sweep-{cfg.sweep_cfg}'
            ]
            # subprocess.run(['./launch_dummy.sh', *init_args, *additional_cli_flags, *meta_flags])
            subprocess.run(['sbatch', 'launch.sh', *init_args, *additional_cli_flags, *meta_flags])
        exit(0)


    logger = logging.getLogger(__name__)
    logger.info(f"Running NDT2, dumping config:")
    logger.info(OmegaConf.to_yaml(cfg))
    pl.seed_everything(seed=cfg.seed)

    dataset = SpikingDataset(cfg.dataset)
    dataset.build_context_index()
    if cfg.dataset.eval_datasets:
        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.subset_split(splits=['eval'], keep_index=True)
    dataset.subset_split(keep_index=True)
    train, val = dataset.create_tv_datasets()
    logger.info(f"Training on {len(train)} examples")
    data_attrs = dataset.get_data_attrs()
    logger.info(pformat(f"Data attributes: {data_attrs}"))

    if cfg.init_from_id:
        init_ckpt = get_best_ckpt_from_wandb_id(
            cfg.wandb_project, cfg.init_from_id,
            tag=cfg.init_tag
        )
        logger.info(f"Initializing from {init_ckpt}")
        model = load_from_checkpoint(init_ckpt, cfg=cfg.model, data_attrs=data_attrs)
    else:
        model = BrainBertInterface(cfg.model, data_attrs)
    if cfg.model.task.freeze_backbone:
        model.freeze_backbone()
    if cfg.model.task.freeze_all:
        model.freeze_non_embed()

    epochs = cfg.train.epochs
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',
            filename='val-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            every_n_epochs=1,
            # every_n_train_steps=cfg.train.val_check_interval,
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
        if reset_early_stop:
            def null_load(*args):
                return
        callbacks[-1].load_state_dict = null_load

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if cfg.model.lr_schedule != "fixed":
        callbacks.append(lr_monitor)

    for m in [Metric.co_bps, Metric.bps]:
        if m in cfg.model.task.metrics:
            callbacks.append(
                ModelCheckpoint(
                    monitor=f'val_{m.value}',
                    filename='val_' + m.value + '-{epoch:02d}-{val_' + m.value + ':.4f}',
                    save_top_k=1,
                    mode='max',
                    every_n_epochs=1,
                    # every_n_train_steps=cfg.train.val_check_interval,
                    dirpath=None
                )
            )

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        save_dir=cfg.default_root_dir,
    )

    pl.seed_everything(seed=cfg.seed)

    if cfg.train.steps:
        max_steps = cfg.train.steps
        epochs = None
    else:
        max_steps = -1

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        check_val_every_n_epoch=1,
        # val_check_interval=cfg.train.val_check_interval,
        callbacks=callbacks,
        default_root_dir=cfg.default_root_dir,
        track_grad_norm=2 if cfg.train.log_grad else -1,
        precision=16 if cfg.model.half_precision else 32,
        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else None,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_batches,
        profiler=cfg.train.profiler if cfg.train.profiler else None,
    )

    if torch.cuda.device_count() <= 1 or trainer.global_rank == 0:
        # Note, wandb.run can also be accessed as logger.experiment but there's no benefit
        if cfg.tag:
            wandb.run.name = f'{cfg.tag}-{wandb.run.id}'
        if os.environ.get('SLURM_JOB_ID'):
            wandb.run.notes = f"SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}"
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update({'data_attrs': dataclasses.asdict(data_attrs)})
        # Of course now I find these
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("eval_loss", summary="min")
        wandb.define_metric(f"val_{Metric.bps.value}", summary="max")
        wandb.define_metric(f"eval_{Metric.bps.value}", summary="max")

    # === Train ===
    # num_workers = len(os.sched_getaffinity(0)) # If this is set too high, the dataloader may crash.
    num_workers = 0 # for testing
    if num_workers == 0:
        logger.warning("Num workers is 0, DEBUGGING.")
    logger.info("Preparing to fit...")

    val_datasets = [val]
    if cfg.dataset.eval_datasets:
        val_datasets.append(eval_dataset)
    data_module = SpikingDataModule(
        cfg.train.batch_size,
        num_workers,
        train, val_datasets
    )
    # import pdb;pdb.set_trace()
    if torch.cuda.device_count() <= 1 and len(train) > 2000 and 'test' not in cfg.tag and getattr(cfg.train, 'autoscale_batch_size', True): # don't scale test debug runs
        new_bsz = trainer.tuner.scale_batch_size(model, datamodule=data_module, mode="power", steps_per_trial=15, max_trials=20)
        data_module.batch_size = new_bsz

    trainer.fit(
        model, datamodule=data_module,
        ckpt_path=get_best_ckpt_from_wandb_id(cfg.wandb_project, cfg.load_from_id) if cfg.load_from_id else None
    )
    logger.info('Run complete')

if __name__ == '__main__':
    run_exp()