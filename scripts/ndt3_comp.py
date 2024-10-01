#%%
from typing import List
import os
# set cuda to device 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Demonstrate pretraining scaling. Assumes evaluation metrics have been computed and merely assembles.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import r2_score
import pandas as pd
import pytorch_lightning as pl

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.config.hp_sweep_space import sweep_space
from context_general_bci.dataset import SpikingDataset
from context_general_bci.model import transfer_model
from context_general_bci.analyze_utils import stack_batch, get_dataloader, load_wandb_run, prep_plt, get_best_ckpt_from_wandb_id, get_run_config
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, to_device


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
    # EVAL_SET = "falcon_h1"
    # EVAL_SET = "falcon_m1"
    EVAL_SET = "cursor"
    # EVAL_SET = "rtt"
    EVAL_SET = "grasp_h"
    SCALES = [0.25, 0.5, 1.0]
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", "-e", type=str, required=True, choices=[
            'falcon_h1',
            'falcon_m1',
            'falcon_h1_patient',
            # 'falcon_m1_continual', # v4 defunct
            'falcon_m2',
            'falcon_m2_patient',
            # 'falcon_m2_continual', # v4 defunct
            'cursor',
            'rtt',
            'grasp_h',
            'bimanual',
            'cst',
            'cursor_new',
            'grasp_new',]
    )
    parser.add_argument(
        "--scales", "-s", type=float, nargs='+', default=[0.03, 0.1, 0.25, 0.5, 1.0]
    )
    args = parser.parse_args()
    EVAL_SET = args.eval_set
    SCALES = args.scales

TAG_MAP = {
    "falcon_h1": "h1_{scale_ratio}",
    "falcon_h1_patient": "h1_{scale_ratio}",
    "falcon_m1": "m1_{scale_ratio}",
    "falcon_m1_continual": "m1_{scale_ratio}_continual",
    "falcon_m2": "m2_{scale_ratio}",
    "falcon_m2_patient": "m2_{scale_ratio}",
    "falcon_m2_continual": "m2_{scale_ratio}_continual",
    "cursor": "cursor_{scale_ratio}",
    "rtt": "rtt_{scale_ratio}",
    "grasp_h": "grasp_{scale_ratio}",
    "bimanual": "bimanual_{scale_ratio}",
    "cst": "cst_{scale_ratio}",
    "cursor_new": "cursor_{scale_ratio}",
    "grasp_new": "grasp_{scale_ratio}",
}
NDT2_EXPERIMENT_MAP = {
    "falcon_h1": "falcon/h1",
    "falcon_h1_patient": "falcon/h1_patient",
    "falcon_m1": "falcon/m1",
    "falcon_m1_continual": "falcon/m1",
    "falcon_m2": "falcon/m2",
    "falcon_m2_patient": "falcon/m2_patient",
    "falcon_m2_continual": "falcon/m2",
    "cursor": "eval/cursor",
    "rtt": "eval/rtt",
    "grasp_h": "eval/grasp_h",
    "bimanual": "eval/bimanual",
    "cst": "eval/cst",
    "cursor_new": "eval/cursor_new",
    "grasp_new": "eval/grasp_new",
}

UNIQUE_BY = {
    "model.lr_init",
    "model.hidden_size",
    "dataset.scale_ratio",
    "seed",
    # "dataset.falcon_m2.respect_trial_boundaries",
}

EVAL_DATASET_FUNC_MAP = {
    'falcon_h1': None, # TODO
    'falcon_h1_patient': None, # TODO
    'falcon_m1': None, # TODO
    'falcon_m1_continual': None, # TODO
    'falcon_m2': None,
    'falcon_m2_patient': None, # TODO
    'falcon_m2_continual': None,
    'cursor': 'eval_pitt_eval_broad.*',
    'rtt': 'eval_odoherty_eval_rtt.*',
    "grasp_h": "eval_pitt_grasp_h.*",
    "bimanual": "deo.*",
    "cst": "cst.*",
    "cursor_new": "cursor_new.*",
    "grasp_new": "grasp_new.*",
}

eval_paths = Path('./data/eval_metrics')
def get_runs_for_query(variant: str, scale_ratio: float, eval_set: str, project="ndt3", exp_map=NDT2_EXPERIMENT_MAP):
    r"""
        variant: init_from_id
    """
    # sweep_tag = "simple_scratch" if "scratch" in variant else "simple_ft" # v4 settings
    sweep_tag = "full_scratch"
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
    if 'continual' in eval_set_name:
        eval_crop = eval_set_name.replace('_continual', '')
    elif 'patient' in eval_set_name:
        eval_crop = eval_set_name.replace('_patient', '')
    else:
        eval_crop = eval_set_name
    if eval_crop in ['cursor', 'grasp_h', 'cursor_new', 'grasp_new']:
        config_key = 'pitt_co'
    elif eval_crop == 'rtt':
        config_key = 'odoherty_rtt'
    elif eval_crop == 'bimanual':
        config_key = 'deo'
    else:
        config_key = eval_crop
    df_dict = {
        'id': map(lambda r: r.id, filter_runs),
        'variant': map(lambda r: r.config['tag'], filter_runs),
        'scale_ratio': map(lambda r: r.config['dataset']['scale_ratio'], filter_runs),
        "respect_trial_boundaries": map(lambda r: r.config['dataset'][config_key].get('respect_trial_boundaries', True), runs),
        'eval_set': map(lambda r: eval_set_name, filter_runs),
        'val_kinematic_r2': map(lambda r: r.summary['val_kinematic_r2']['max'], filter_runs),
        'sweep': ['full_scratch'] * len(filter_runs),
    }
    def nested_get_from_config(config, param: List[str]):
        if len(param) > 1:
            return nested_get_from_config(config[param[0]], param[1:])
        return config[param[0]]
    unique_sweeps = set(df_dict['sweep'])
    for sweep_name in unique_sweeps:
        for p in sweep_space[sweep_name].keys():
            # For some reason if we don't cast, early params get overwritten..
            df_dict[p] = list(map(lambda r: nested_get_from_config(r.config, p.split('.')), filter_runs))
    if eval_set_name in ['rtt', 'cursor', 'grasp_h']: # Not separate pipeline
        df_dict['eval_report'] = map(lambda r: r.summary['eval_kinematic_r2']['max'], runs)
    return pd.DataFrame(df_dict)

def get_best_run_per_sweep(run_df, metric='val_kinematic_r2'):
    if 'seed' in run_df:
        hp_columns = [col for col in run_df.columns if col not in ['id', 'variant', 'eval_set', 'scale_ratio', 'seed', 'val_kinematic_r2', 'eval_report']]
        id_columns = ['variant']
        group_columns = [*hp_columns, *id_columns]
        seed_averaged_df = run_df.groupby(group_columns)[metric].mean().reset_index()
        aug_df = pd.merge(run_df, seed_averaged_df, on=group_columns, suffixes=('', '_seed_avg'))
        filter_metric = f'{metric}_seed_avg'
        run_df = aug_df.groupby('variant').apply(lambda x: x[x[filter_metric] == x[filter_metric].max()]).reset_index(drop=True)
    else: # Then re-group by variant and filter for the best HP.
        run_df = run_df.groupby(['variant']).apply(lambda x: x.nlargest(1, metric)).reset_index(drop=True)
    return run_df

def get_run_df_for_query(variant: str, scale_ratio: float, eval_set: str, metric='val_kinematic_r2', **kwargs):
    runs = get_runs_for_query(variant, scale_ratio, eval_set, **kwargs)
    run_df = run_list_to_df(runs, eval_set)
    return get_best_run_per_sweep(run_df, metric=metric)

def get_ndt2_run_df_for_query(scale_ratio: float, eval_set: str):
    # only "scratch" runs compared for ndt2
    return get_run_df_for_query("scratch", scale_ratio, eval_set, project="context_general_bci", exp_map=NDT2_EXPERIMENT_MAP)

ndt2_run_df = pd.concat([get_ndt2_run_df_for_query(scale_ratio, EVAL_SET) for scale_ratio in SCALES]).reset_index()
# ndt2_run_df = pd.concat([get_ndt2_run_df_for_query(scale_ratio, EVAL_SET) for scale_ratio in SCALE_MAP[EVAL_SET]]).reset_index()
eval_metrics_path = eval_paths / f"{EVAL_SET}_eval_ndt2.csv"
def load_eval_df_so_far(eval_metrics_path):
    return pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()
eval_df_so_far = load_eval_df_so_far(eval_metrics_path)

def trim_df(df, df_so_far):
    # Delete the data from eval queue that already exists in so_far
    if len(df_so_far):
        if 'index' in df_so_far:
            df_so_far.drop(columns=['index'], inplace=True)
        # df_so_far zero to nan
        df_so_far['eval_r2'] = df_so_far['eval_r2'].replace(0, np.nan)
        # df_so_far drop nan
        df_so_far = df_so_far.dropna(subset=['eval_r2'])
        df = df[~df.id.isin(df_so_far.id)].reset_index(drop=True)
    return df

eval_metrics = {}
def get_single_eval(cfg: RootConfig, src_model, trainer, dataset=None):
    pl.seed_everything(0)
    if dataset is None:
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    cfg.model.task.tasks = [ModelTask.kinematic_decoding]
    model = transfer_model(src_model, cfg.model, data_attrs)
    model = model.to('cuda')
    if EVAL_SET == 'cursor':
        batch_size = 64
    elif EVAL_SET == 'grasp_h':
        batch_size = 8
    elif EVAL_SET == 'bimanual':
        batch_size = 64
    else: # RTT
        batch_size = 256

    model.cfg.task.outputs = [Output.behavior, Output.behavior_pred]
    if EVAL_SET not in ['eye', 'cst', 'rtt', 'bimanual']:
        if 'grasp' in EVAL_SET:
            stream_buffer_s = 2.
        else:
            stream_buffer_s = 1 # streaming eval, mirror NDT3
        # TODO streaming eval
        # NDT3's streaming eval implementation relies on KV cache, which is not applicable here.
        # Here we will assume limited size evaluation dataset and just load everything at once, then slide through.
        # NDT2 batch must be spoofed - raw data doesn't flatten (i.e. we could've concated on dim 0), but batches do.
        # We thus either need to spoof batching or spoof unflattening/sliding logic, latter is chosen here.
        dataloader = get_dataloader(dataset, num_workers=0, batch_size=1)
        accum_batch = {
            DataKey.spikes: [],
            DataKey.time: [],
            DataKey.position: [],
            DataKey.bhvr_vel: [],
            DataKey.bhvr_mask: [],
            'channel_counts': []
        }
        meta_batch = {}

        running_start_time = 0
        for i, batch in enumerate(dataloader):
            for key in batch:
                if key in accum_batch:
                    if key == DataKey.time:
                        accum_batch[key].append(batch[key] + running_start_time)
                        running_start_time += batch[key].max() + 1
                    else:
                        accum_batch[key].append(batch[key])
                elif key not in meta_batch:
                    meta_batch[key] = batch[key]
        for key in accum_batch:
            accum_batch[key] = torch.cat(accum_batch[key], dim=1)

        timesteps = accum_batch[DataKey.time].max()
        pred, true, masks = [], [], []
        for end_time_inclusive in tqdm(range(timesteps)):
            start_timestep = max(0, int(end_time_inclusive - stream_buffer_s * 1000 / cfg.dataset.bin_size_ms))
            sliding_batch = {k: accum_batch[k][(accum_batch[DataKey.time] <= end_time_inclusive) & (accum_batch[DataKey.time] >= start_timestep)]
                             for k in [DataKey.time, DataKey.spikes, DataKey.position, 'channel_counts']}
            for k in [DataKey.bhvr_vel, DataKey.bhvr_mask]:
                if k in accum_batch:
                    sliding_batch[k] = accum_batch[k][:, start_timestep:end_time_inclusive+1]
            sliding_batch[DataKey.time] = sliding_batch[DataKey.time] - start_timestep
            sliding_batch = {**sliding_batch, **meta_batch}
            sliding_batch = to_device(sliding_batch, device='cuda')
            outputs = model.predict(sliding_batch)
            # Should have a leading batch of 1, since dataloader has batch size 1, remove and take final timestep
            pred.append(outputs[Output.behavior_pred][0, -1:].cpu()) # B T H -> T=1 H
            true.append(outputs[Output.behavior][0, -1:].cpu()) # B T H -> T=1 H
            masks.append(outputs[Output.behavior_mask][0, -1:].cpu()) # B T -> T=1
        padding = None
    else:
        stream_buffer_s = 0
        dataloader = get_dataloader(dataset, num_workers=4, batch_size=batch_size)
        trainer_preds = trainer.predict(model, dataloader)
        outputs = stack_batch(trainer_preds) # Trial x Time x Dim, first dim may be list
        pred, true = outputs[Output.behavior_pred], outputs[Output.behavior]
        masks = outputs[Output.behavior_mask] if Output.behavior_mask in outputs else None
        if Output.padding in outputs:
            padding = outputs[Output.padding]
        else:
            padding = None
    if isinstance(pred, list):
        pred = torch.cat(pred, dim=0)
        true = torch.cat(true, dim=0)
        if masks is not None:
            masks = torch.cat(masks, dim=0)
            if masks.ndim != true.ndim - 1:
                masks = masks.squeeze(-1) # remove hidden dim of 1
        if padding is not None:
            padding = torch.cat(padding, dim=0)
            pred = pred[~padding]
            true = true[~padding]
            masks = masks[~padding] if masks is not None else None
    else:
        pred = pred.flatten(end_dim=-2)
        true = true.flatten(end_dim=-2)
        if masks is not None:
            masks = masks.flatten(end_dim=-2).squeeze(-1)
    if masks is not None:
        pred = pred[masks]
        true = true[masks]
    r2 = r2_score(true, pred, multioutput='variance_weighted')
    print(f"R2 over {len(pred)} samples: {r2:.3f}")
    return r2

def commit_df(df, path):
    df_so_far = load_eval_df_so_far(path)
    df = pd.concat([df, df_so_far]).reset_index(drop=True)
    df.to_csv(path, index=False)
    df = df.drop_duplicates(subset=['variant', 'seed'], keep='first').reset_index(drop=True)
    df.to_csv(path, index=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
ndt2_run_df = trim_df(ndt2_run_df, eval_df_so_far)
ndt2_run_df['eval_r2'] = 0.
for idx, run_row in ndt2_run_df.iterrows():
    eval_set = EVAL_DATASET_FUNC_MAP[run_row.eval_set]
    run = get_wandb_run(run_row.id, wandb_project="context_general_bci")
    if eval_set is not None:
        print('\n', run, '\n')
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_kinematic_r2')
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        dataset.subset_split(splits=['eval'])
        eval_r2 = get_single_eval(cfg, src_model, trainer, dataset=dataset)
        ndt2_run_df.at[idx, 'eval_r2'] = eval_r2  # Correct way to modify a DataFrame row
        commit_df(ndt2_run_df, eval_metrics_path)
    elif 'falcon' in run_row.eval_set:
        cfg = get_run_config(run, tag='val_kinematic_r2')
        ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag='val_kinematic_r2')
        zscore_pth = cfg.model.task.decode_normalizer
        split = run_row.eval_set.split('_')[1]
        os.environ['GT_PATH'] = f'./local_gt_{split}.pkl'
        os.environ['PREDICTION_PATH_LOCAL'] = f'./local_pred_{run_row.id}_{split}.pkl'
        evaluator = FalconEvaluator(
            eval_remote=False,
            split=split,
        )
            # continual='continual' in run_row.variant)

        task = getattr(FalconTask, split)
        config = FalconConfig(task=task)
        # if '_continual' in run_row.variant:
        #     max_bins = cfg.dataset.augment_crop_length_ms // config.bin_size_ms
        # else:
        #     max_bins = cfg.dataset.max_length_ms // config.bin_size_ms
        if split == 'h1':
            max_bins = 200 # 4s (V5 NDT3 comp)
        else:
            max_bins = 50 # 1s (V5 NDT3 comp)
        decoder = NDT2Decoder(
            task_config=config,
            model_ckpt_path=ckpt,
            model_cfg_stem=f'{cfg.experiment_set}/{cfg.tag.split("-")[0]}',
            zscore_path=zscore_pth,
            max_bins=max_bins,
            dataset_handles=[x.stem for x in evaluator.get_eval_files(phase='test')],
            batch_size=8 if 'continual' in run_row.variant else 1,
        )
        payload = evaluator.evaluate(decoder, phase='test')
        eval_r2 = payload['result'][0][f'test_split_{split}']['Held Out R2 Mean']
        if 'heldin_eval_r2' not in ndt2_run_df.columns:
            ndt2_run_df['heldin_eval_r2'] = 0.
        heldin_eval_r2 = payload['result'][0][f'test_split_{split}']['Held In R2 Mean']
        ndt2_run_df.at[idx, 'eval_r2'] = eval_r2
        ndt2_run_df.at[idx, 'heldin_eval_r2'] = heldin_eval_r2
        print(ndt2_run_df.iloc[idx])

#%%
print(ndt2_run_df)
import seaborn as sns
ax = prep_plt()

sns.lineplot(data=ndt2_run_df, x='scale_ratio', y='eval_r2', hue='eval_set', ax=ax)

# save down
commit_df(ndt2_run_df, eval_metrics_path)