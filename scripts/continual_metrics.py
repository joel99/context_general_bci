#%%
# Compute oracle FALCON metrics by identifying the best run in a group
import os
# set cuda to device 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.analyze_utils import stack_batch, get_dataloader, load_wandb_run, prep_plt, get_best_ckpt_from_wandb_id, get_run_config
from context_general_bci.utils import wandb_query_experiment, get_wandb_run


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
    VARIANT = 'm2'
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", "-v", type=str, required=True, choices=[
            'h1', 
            'm1', 
            'm2',
        ]
    )
    parser.add_argument(
        "--trialized", "-t", action='store_true'
    )
    args = parser.parse_args()
    VARIANT = args.variant
    TRIALIZED_EVAL = args.trialized # still requires hardcode override in package - manually do this
eval_set = VARIANT

TAG_MAP = {
    "h1": "h1_chop",
    "m1": "m1_chop",
    "m2": "m2_chop",
}

NDT2_EXPERIMENT_MAP = {
    "h1": "falcon/h1",
    "m1": "falcon/m1",
    "m2": "falcon/m2",
}

UNIQUE_BY = {
    "model.lr_init", 
    "dataset.falcon_m2.chop_size_ms",
    "dataset.falcon_m1.chop_size_ms",
    "dataset.falcon_h1.chop_size_ms",
    "dataset.augment_crop_length_ms",
    "model.hidden_size", 
    "dataset.scale_ratio",
    "dataset.datasets",
    "dataset.falcon_m2.respect_trial_boundaries",
}

eval_paths = Path('./data/falcon_metrics')
eval_paths.mkdir(exist_ok=True, parents=True)

def get_runs_for_query(variant: str, project="context_general_bci", exp_map=NDT2_EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    variant_tag = TAG_MAP[variant]
    print(f'Querying: {variant_tag}')
    return wandb_query_experiment(
        exp_map[variant], 
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant_tag},
            "config.sweep_tag": {"$in": ['chop_coarse_4s', 'chop_coarse_2s', 'chop_coarse_1s', 'chop_coarse_500ms']},
            "state": {"$in": ['finished', 'crashed']},
        })

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    # Attempt to recover runs if no explicit summary - we only need r2...
    filter_runs
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    for i in filter_runs:
        print(i)
    eval_crop = f'falcon_{eval_set_name}'
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'lr_init': map(lambda r: r.config['model']['lr_init'], filter_runs),
        "chop_size_ms": map(lambda r: r.config['dataset'][eval_crop].get('chop_size_ms', 0), filter_runs),
        "augment_chop_length_ms": map(lambda r: r.config['dataset'].get('augment_crop_length_ms', 0), filter_runs),
        "respect_trial_boundaries": map(lambda r: r.config['dataset'][eval_crop].get('respect_trial_boundaries', True), filter_runs),
        'eval_set': map(lambda r: eval_set_name, filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
    }
    return pd.DataFrame(df_dict)

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    # reduce the df to only the runs with the highest R2
    run_df = run_df.groupby(['augment_chop_length_ms']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    # run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

def get_ndt2_run_df_for_query(variant: str):
    return get_run_df_for_query(variant, eval_set)

queries = [
    ""
]

runs = []

ndt2_run_df = get_ndt2_run_df_for_query(VARIANT)

print(ndt2_run_df)

eval_metrics_path = eval_paths / f"{eval_set}_{'trialized' if TRIALIZED_EVAL else 'continual'}_ndt2.csv"
eval_df_so_far = pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()

ndt2_run_df = pd.concat([ndt2_run_df, eval_df_so_far]).reset_index(drop=True)
if len(eval_df_so_far):
    if 'index' in eval_df_so_far:
        eval_df_so_far.drop(columns=['index'], inplace=True)
    # eval_df_so_far zero to nan
    eval_df_so_far['eval_r2'] = eval_df_so_far['eval_r2'].replace(0, np.nan)
    # eval_df_so_far drop nan
    eval_df_so_far = eval_df_so_far.dropna(subset=['eval_r2'])
    ndt2_run_df = ndt2_run_df[~ndt2_run_df.id.isin(eval_df_so_far.id)].reset_index(drop=True)

eval_metrics = {}
ndt2_run_df['eval_r2'] = 0.
for idx, run_row in ndt2_run_df.iterrows():
    run = get_wandb_run(run_row.id, wandb_project="context_general_bci")
    cfg = get_run_config(run, tag='val_kinematic_r2')
    ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag='val_kinematic_r2')
    zscore_pth = cfg.model.task.decode_normalizer
    split = run_row.eval_set
    if split == 'm2':
        zscore_pth = "" # Forgot to provide this but training worked fine
    breakpoint()
    evaluator = FalconEvaluator(
        eval_remote=False,
        split=split,
        verbose=True
    )

    task = getattr(FalconTask, split)
    config = FalconConfig(task=task)
    max_bins = cfg.dataset.augment_crop_length_ms // config.bin_size_ms
    decoder = NDT2Decoder(
        task_config=config,
        model_ckpt_path=ckpt,
        model_cfg_stem=f'{cfg.experiment_set}/{cfg.tag.split("-")[0]}',
        zscore_path=zscore_pth,
        max_bins=max_bins,
        dataset_handles=[x.stem for x in evaluator.get_eval_files(phase='test')],
        batch_size=1 if TRIALIZED_EVAL else 8,
    )
    payload = evaluator.evaluate(decoder, phase='test')
    result = payload['result'][0][f'test_split_{split}']
    eval_r2 = result['Held Out R2 Mean']
    if 'heldin_eval_r2' not in ndt2_run_df.columns:
        ndt2_run_df['heldin_eval_r2'] = 0.
    heldin_eval_r2 = result['Held In R2 Mean']
    ndt2_run_df.at[idx, 'eval_r2'] = eval_r2
    ndt2_run_df.at[idx, 'heldin_eval_r2'] = heldin_eval_r2
    # Copy down all other metrics
    del result['Held Out R2 Mean']
    del result['Held In R2 Mean']
    for key, val in result.items():
        ndt2_run_df.at[idx, key] = val
    print(ndt2_run_df.iloc[idx])

#%%
ndt2_run_df = pd.concat([ndt2_run_df, eval_df_so_far]).reset_index(drop=True)
# drop duplicates by variant stem, prefer new
ndt2_run_df = ndt2_run_df.drop_duplicates(subset=['variant'], keep='first').reset_index(drop=True)

ndt2_run_df.to_csv(eval_metrics_path, index=False)
