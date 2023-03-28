import os
import sys
from pathlib import Path
import copy
import subprocess
import functools

from typing import Dict, Any

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
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb

from config import RootConfig, Metric, hp_sweep_space
from data import SpikingDataset, SpikingDataModule
from model import BrainBertInterface, load_from_checkpoint
from callbacks import ProbeToFineTuneEarlyStopping
from analyze_utils import get_best_ckpt_from_wandb_id
from utils import generate_search, grid_search

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
    - The latter runs but doesn't log `experiment_set` correctly
"""
reset_early_stop = True # todo move into config

@rank_zero_only
def init_wandb(cfg, wandb_logger):
    # if wandb.run == None:
    #     wandb.init(project=cfg.wandb_project) # for some reason wandb changed and now I need a declaration
    _ = wandb_logger.experiment # force experiment recognition so that id is initialized

@hydra.main(version_base=None, config_path='config', config_name="config")
def run_exp(cfg : RootConfig) -> None:
    # Check for sweeping. Note we process data above because I never intend to sweep over data config.
    if cfg.tag == "":
        r"""
            JY is used to having experimental variant names tracked with filename (instead of manually setting tags)
            take sys.argv and set the tag to the query. Only set it if we're not sweeping (where tag was already explicitly set)
        """
        exp_arg = [arg for arg in sys.argv if '+exp' in arg]
        if len(exp_arg) > 0:
            cfg.tag = exp_arg[0].split('=')[1]
            cfg.experiment_set = exp_arg[0].split('=')[0][len('+exp/'):]
    if cfg.sweep_cfg and os.environ.get('SLURM_JOB_ID') is None: # do not allow recursive launch
        sweep_cfg = hp_sweep_space.sweep_space[cfg.sweep_cfg]
        def run_cfg(cfg_trial):
            init_call = sys.argv
            init_args = init_call[init_call.index('run.py')+1:]
            additional_cli_flags = [f'{k}={v}' for k, v in cfg_trial.items()]
            meta_flags = [
                'sweep_cfg=""',
                f'sweep_tag={cfg.sweep_cfg}',
                f'tag={cfg.tag}-sweep-{cfg.sweep_cfg}',
                f'experiment_set={cfg.experiment_set}'
            ]
            # subprocess.run(['./launch_dummy.sh', *init_args, *additional_cli_flags, *meta_flags])
            # subprocess.run(['sbatch', './crc_scripts/launch.sh', *init_args, *additional_cli_flags, *meta_flags])
            subprocess.run(['sbatch', 'launch.sh', *init_args, *additional_cli_flags, *meta_flags])
        if cfg.sweep_mode == 'grid':
            # Create a list of dicts from the cross product of the sweep config
            for cfg_trial in grid_search(sweep_cfg):
                run_cfg(cfg_trial)
        else:
            for cfg_trial in generate_search(sweep_cfg, cfg.sweep_trials):
                run_cfg(cfg_trial)
        exit(0)

    logger = logging.getLogger(__name__)
    pl.seed_everything(seed=cfg.seed)

    dataset = SpikingDataset(cfg.dataset)
    dataset.build_context_index()
    if cfg.dataset.eval_datasets:
        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.subset_split(splits=['eval'], keep_index=True)
    dataset.subset_split(keep_index=True)
    if cfg.dataset.scale_limit_per_session or cfg.dataset.scale_limit_per_eval_session:
        dataset.subset_scale(
            limit_per_session=cfg.dataset.scale_limit_per_session,
            limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
            keep_index=True
        )
    elif cfg.dataset.scale_ratio:
        dataset.subset_scale(ratio=cfg.dataset.scale_ratio, keep_index=True)
    train, val = dataset.create_tv_datasets()
    logger.info(f"Training on {len(train)} examples")
    data_attrs = dataset.get_data_attrs()
    # logger.info(pformat(f"Data attributes: {data_attrs}"))

    if cfg.init_from_id:
        init_ckpt = get_best_ckpt_from_wandb_id(
            cfg.wandb_project, cfg.init_from_id,
            tag=cfg.init_tag
        )
        logger.info(f"Initializing from {init_ckpt}")
        model = load_from_checkpoint(init_ckpt, cfg=cfg.model, data_attrs=data_attrs)
    else:
        model = BrainBertInterface(cfg.model, data_attrs)
    if cfg.model.task.freeze_embed:
        model.freeze_embed()
    if cfg.model.task.freeze_backbone:
        model.freeze_backbone()
    if cfg.model.task.freeze_all:
        model.freeze_non_embed()

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
        early_stop_cls = ProbeToFineTuneEarlyStopping if cfg.probe_finetune else EarlyStopping
        callbacks.append(
            early_stop_cls(
                monitor='val_loss',
                patience=cfg.train.patience, # Learning can be fairly slow, larger patience should allow overfitting to begin (which is when we want to stop)
                min_delta=1e-5,
            )
        )
        if not cfg.probe_finetune and reset_early_stop:
            def patient_load(self, state_dict: Dict[str, Any]): # would use something more subtle but IDK how to access self
                self.wait_count = 0
                self.stopped_epoch = state_dict["stopped_epoch"]
                self.best_score = state_dict["best_score"]
                self.patience = state_dict["patience"]
            callbacks[-1].load_state_dict = functools.partial(patient_load, callbacks[-1])

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

    pl.seed_everything(seed=cfg.seed)

    if cfg.train.steps:
        max_steps = cfg.train.steps
        epochs = None
    else:
        max_steps = -1
        epochs = cfg.train.epochs

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        save_dir=cfg.default_root_dir,
    )

    init_wandb(cfg, wandb_logger) # needed for checkpoint to save under wandb dir, for some reason wandb api changed.

    is_distributed = (torch.cuda.device_count() > 1) or getattr(cfg, 'nodes', 1) > 1

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        max_steps=max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        num_nodes=getattr(cfg, 'nodes', 1),
        check_val_every_n_epoch=1,
        log_every_n_steps=cfg.train.log_every_n_steps,
        # val_check_interval=cfg.train.val_check_interval,
        callbacks=callbacks,
        default_root_dir=cfg.default_root_dir,
        track_grad_norm=2 if cfg.train.log_grad else -1, # this is quite cluttered, but probably better that way. See https://github.com/Lightning-AI/lightning/issues/1462#issuecomment-1190253742 for patch if needed, though.
        precision=16 if cfg.model.half_precision else 32,
        strategy=DDPStrategy(find_unused_parameters=False) if is_distributed else None,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_batches,
        profiler=cfg.train.profiler if cfg.train.profiler else None,
        overfit_batches=1 if cfg.train.overfit_batches else 0
    )


    # Note, wandb.run can also be accessed as logger.experiment but there's no benefit
    # torch.cuda.device_count() > 1 or cfg.nodes > 1
    if trainer.global_rank == 0:
        logger.info(f"Running NDT2, dumping config:")
        logger.info(OmegaConf.to_yaml(cfg))
        if cfg.tag:
            wandb.run.name = f'{cfg.tag}-{wandb.run.id}'
        notes = cfg.notes
        if os.environ.get('SLURM_JOB_ID'):
            wandb.run.notes = f"{notes}. SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}"
            cfg.slurm_id = int(os.environ['SLURM_JOB_ID'])
            if not cfg.train.autoscale_batch_size:
                cfg.effective_bsz = cfg.train.batch_size * cfg.train.accumulate_batches * cfg.nodes * torch.cuda.device_count()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update({'data_attrs': dataclasses.asdict(data_attrs)})
        # Of course now I find these
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("eval_loss", summary="min")
        wandb.define_metric(f"val_{Metric.bps.value}", summary="max")
        wandb.define_metric(f"eval_{Metric.bps.value}", summary="max")
        wandb.define_metric(f"val_{Metric.kinematic_r2.value}", summary="max")
        wandb.define_metric(f"eval_{Metric.kinematic_r2.value}", summary="max")

    # === Train ===
    num_workers = len(os.sched_getaffinity(0)) # If this is set too high, the dataloader may crash.
    # num_workers = 0 # for testing
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

    if not is_distributed and cfg.train.autoscale_batch_size: # autoscale doesn't work for DDP
        new_bsz = trainer.tuner.scale_batch_size(model, datamodule=data_module, mode="power", steps_per_trial=15, max_trials=20)
        if cfg.train.max_batch_size:
            new_bsz = min(new_bsz, cfg.train.max_batch_size)
        data_module.batch_size = new_bsz
    trainer.fit(
        model, datamodule=data_module,
        ckpt_path=get_best_ckpt_from_wandb_id(cfg.wandb_project, cfg.load_from_id) if cfg.load_from_id else None
    )
    logger.info('Run complete')

if __name__ == '__main__':
    run_exp()