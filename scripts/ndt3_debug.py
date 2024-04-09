#%%
import os
# set cuda to device 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
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


from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from context_general_bci.falcon_decoder import NDT2Decoder
# from scripts.falcon_local import run_evaluate

pl.seed_everything(0)

EVAL_SET = "falcon_m1"

TAG_MAP = {
    "falcon_h1": "h1_{scale_ratio}",
    "falcon_m1": "m1_{scale_ratio}",
    "cursor": "cursor_{scale_ratio}",
    "rtt": "rtt_{scale_ratio}",
}
NDT2_EXPERIMENT_MAP = {
    "falcon_h1": "falcon/h1",
    "falcon_m1": "falcon/m1",
    "cursor": "eval/cursor",
    "rtt": "eval/rtt",
}

EXPERIMENT_MAP = {
    "falcon_h1": "v4/tune/falcon_h1",
    "falcon_m1": "v4/tune/falcon_m1",
    "cursor": "v4/tune/cursor",
    "rtt": "v4/tune/rtt",
}

UNIQUE_BY = {
    "model.lr_init", 
    "model.hidden_size", 
    "dataset.scale_ratio",
}

EVAL_DATASET_FUNC_MAP = {
    'falcon_h1': None, # TODO
    'falcon_m1': None, # TODO
    'cursor': 'eval_pitt_eval_broad.*',
    'rtt': 'eval_odoherty_eval_rtt.*'
}

SCALE_MAP = {
    'falcon_h1': [0.25, 0.5, 1.0],
    'falcon_m1': [1.0],
    # 'falcon_m1': [0.25, 0.5, 1.0],
    'cursor': [0.25, 0.5, 1.0],
    'rtt': [0.25, 0.5, 1.0],
}

eval_paths = Path('./data/eval_metrics')
def get_runs_for_query(variant: str, scale_ratio: float, eval_set: str, project="ndt3", exp_map=EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    sweep_tag = "simple_scratch" if "scratch" in variant else "simple_ft" # v4 settings
    variant_tag = TAG_MAP[eval_set].format(scale_ratio=int(scale_ratio * 100))
    print(f'Querying: {variant_tag}')
    return wandb_query_experiment(
        exp_map[eval_set], 
        wandb_project=project,
        filter_unique_keys=UNIQUE_BY,
        **{
            "config.tag": {"$regex": variant_tag},
            # "display_name": {"$regex": variant},
            "config.dataset.scale_ratio": scale_ratio,
            "config.sweep_tag": sweep_tag,
            "state": {"$in": ['finished', 'crashed']},
        })

def run_list_to_df(runs, eval_set_name: str):
    filter_runs = [r for r in runs if 'val_kinematic_r2' in r.summary and 'max' in r.summary['val_kinematic_r2']]
    print(f"Filtered {len(runs)} to {len(filter_runs)} with proper metrics")
    df_dict = {
        'id': map(lambda r: r.id, runs),
        'variant': map(lambda r: r.config['tag'], runs),
        'scale_ratio': map(lambda r: r.config['dataset']['scale_ratio'], runs),
        'eval_set': map(lambda r: eval_set_name, runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], runs),
    }
    if eval_set_name in ['rtt', 'cursor']: # Not separate pipeline
        df_dict['eval_report'] = map(lambda r: r.summary['eval_kinematic_r2']['max'], runs)
    return pd.DataFrame(df_dict)

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    # reduce the df to only the runs with the highest R2
    run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, scale_ratio: float, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, scale_ratio, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

def get_ndt2_run_df_for_query(scale_ratio: float, eval_set: str):
    # only "scratch" runs compared for ndt2
    return get_run_df_for_query("scratch", scale_ratio, eval_set, project="context_general_bci", exp_map=NDT2_EXPERIMENT_MAP)

queries = [
    "scratch",
]

runs = []
ndt2_run_df = pd.concat([get_ndt2_run_df_for_query(scale_ratio, EVAL_SET) for scale_ratio in SCALE_MAP[EVAL_SET]]).reset_index()
eval_metrics = {}
def get_single_eval(cfg: RootConfig, src_model, trainer, dataset=None):
    pl.seed_everything(0)
    if dataset is None:
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    dataset.set_crop_mode(1) # Flush start
    breakpoint()
    cfg.model.task.tasks = [ModelTask.kinematic_decoding]
    model = transfer_model(src_model, cfg.model, data_attrs)
    print(f"Eval on {len(dataset)} samples")
    dataloader = get_dataloader(dataset, num_workers=4, batch_size=256 if EVAL_SET != 'cursor' else 64)
    # dataloader = get_dataloader(dataset, num_workers=4, batch_size=256)
    model.cfg.task.outputs = [Output.behavior, Output.behavior_pred]
    trainer_preds = trainer.predict(model, dataloader)
    outputs = stack_batch(trainer_preds) # Trial x Time x Dim, first dim may be list
    pred, true, masks = outputs[Output.behavior_pred], outputs[Output.behavior], outputs[Output.behavior_mask]
    breakpoint()
    if Output.padding in outputs:
        padding = outputs[Output.padding]
    else:
        padding = None
    if isinstance(pred, list):
        pred = torch.cat(pred, dim=0)
        true = torch.cat(true, dim=0)
        masks = torch.cat(masks, dim=0).squeeze(-1)
        if padding is not None:
            padding = torch.cat(padding, dim=0)
        pred = pred[~padding]
        true = true[~padding]
        masks = masks[~padding]
    else:
        pred = pred.flatten(end_dim=-2)
        true = true.flatten(end_dim=-2)
        masks = masks.flatten(end_dim=-2).squeeze(-1)
    pred = pred[masks]
    true = true[masks]
    r2 = r2_score(true, pred, multioutput='variance_weighted')
    uniform_r2 = r2_score(true, pred, multioutput='uniform_average')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    print(f"Uniform R2: {uniform_r2:.3f}")
    return r2, pred, true

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
ndt2_run_df['eval_r2'] = 0.

# Do validation runs
for idx, run_row in ndt2_run_df.iterrows():
    # breakpoint()
    eval_set = EVAL_DATASET_FUNC_MAP[run_row.eval_set]
    # run = get_wandb_run(run_row.id, wandb_project="context_general_bci")
    run = get_wandb_run('m1_nokey-sweep-simple_scratch-asa6gu7s', wandb_project="context_general_bci") # hardcoded eval
    src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2')
    cfg.dataset.datasets = ['falcon_FALCONM1.*eval']
    cfg.dataset.eval_datasets = []
    dataset = SpikingDataset(cfg.dataset, use_augment=False)
    dataset.subset_split()
    # train, val = dataset.create_tv_datasets()
    # dataset = val
    eval_r2, pred, true = get_single_eval(cfg, src_model, trainer, dataset=dataset)
    ndt2_run_df.at[idx, 'eval_r2'] = eval_r2  # Correct way to modify a DataFrame row

print(ndt2_run_df)
#%%