#%%
# Evaluate the presence of overfitting to trial structure by comparing performance on continual eval of different continually trained models.

import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import r2_score
import pandas as pd
import pytorch_lightning as pl

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelTask, Metric, Output, DataKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.model import transfer_model
from context_general_bci.analyze_utils import stack_batch, get_dataloader, load_wandb_run, prep_plt, get_best_ckpt_from_wandb_id, get_run_config
from context_general_bci.utils import wandb_query_experiment, get_wandb_run
from context_general_bci.config.hp_sweep_space import sweep_space

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from context_general_bci.falcon_decoder import NDT2Decoder
# from scripts.falcon_local import run_evaluate

pl.seed_everything(0)

# argparse the eval set
import sys
import argparse

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    EVAL_SET = "m1"
    EVAL_SET = "m2"
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", "-e", type=str, required=True, choices=['m1', 'm2']
    )
    args = parser.parse_args()
    EVAL_SET = args.eval_set

NDT2_EXPERIMENT_MAP = {
    "m1": "falcon/m1",
    "m2": "falcon/m2",
}

UNIQUE_BY = {
    "model.lr_init", 
    "model.hidden_size", 
    "dataset.augment_crop_length_ms",
}

HP_MAP = {
    'm1': sweep_space['chop']['dataset.augment_crop_length_ms']['feasible_points'],
    'm2': sweep_space['chop']['dataset.augment_crop_length_ms']['feasible_points'],
}

eval_paths = Path('./data/falcon_continual')
def get_runs_for_query(variant: str, crop_length_ms: float, eval_set: str, exp_map=None, project="context_general_bci"):
    r"""
        variant: init_from_id
    """
    sweep_tag = "chop"
    print(f'Querying: {variant} {crop_length_ms}')
    return wandb_query_experiment(
        exp_map[eval_set], 
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant},
            # "display_name": {"$regex": variant},
            "config.dataset.augment_crop_length_ms": crop_length_ms,
            "config.sweep_tag": sweep_tag,
            "state": {"$in": ['finished', 'crashed']},
        })

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, runs),
        'variant': map(lambda r: r.config['tag'], runs),
        'augment_chop_length_ms': map(lambda r: r.config['dataset']['augment_crop_length_ms'], runs),
        'eval_set': map(lambda r: eval_set_name, runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], runs),
    }
    return pd.DataFrame(df_dict)

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    # reduce the df to only the runs with the highest R2
    run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, crop_length_ms: float, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, crop_length_ms, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

def get_ndt2_run_df_for_query(crop_length_ms: float, eval_set: str):
    # only "scratch" runs compared for ndt2
    return get_run_df_for_query(f"{eval_set}_chop", crop_length_ms, eval_set, project="context_general_bci", exp_map=NDT2_EXPERIMENT_MAP)

runs = []
ndt2_run_df = pd.concat([get_ndt2_run_df_for_query(hp, EVAL_SET) for hp in HP_MAP[EVAL_SET]]).reset_index()

#%%
eval_metrics = {}
ndt2_run_df['eval_r2'] = 0.
for idx, run_row in ndt2_run_df.iterrows():
    run = get_wandb_run(run_row.id, wandb_project="context_general_bci")
    cfg = get_run_config(run, tag='val_kinematic_r2')
    ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag='val_kinematic_r2')
    zscore_pth = cfg.model.task.decode_normalizer
    split = cfg.experiment_set.split('/')[-1]
    evaluator = FalconEvaluator(
        eval_remote=False,
        split=split,
        continual=True,
    )

    task = getattr(FalconTask, split)
    config = FalconConfig(task=task)

    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=ckpt,
        model_cfg_stem=f'{cfg.experiment_set}/{cfg.tag.split("-")[0]}',
        zscore_path=zscore_pth,
        dataset_handles=[x.stem for x in evaluator.get_eval_files(phase='test')],
        batch_size=8
    )
    payload = evaluator.evaluate(decoder, phase='test')
    eval_r2 = payload['result'][0][f'test_split_{split}']['Held Out R2']
    if 'heldin_eval_r2' not in ndt2_run_df.columns:
        ndt2_run_df['heldin_eval_r2'] = 0.
    heldin_eval_r2 = payload['result'][0][f'test_split_{split}']['Held In R2']
    ndt2_run_df.at[idx, 'eval_r2'] = eval_r2
    ndt2_run_df.at[idx, 'heldin_eval_r2'] = heldin_eval_r2
    print(ndt2_run_df.iloc[idx])

#%%
# print(ndt2_run_df)
# import seaborn as sns
# ax = prep_plt()

# sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='eval_r2', hue='eval_set', ax=ax)

# save down
eval_metrics_path = eval_paths / f"{EVAL_SET}_eval_ndt2.csv"
ndt2_run_df.to_csv(eval_metrics_path, index=False)
