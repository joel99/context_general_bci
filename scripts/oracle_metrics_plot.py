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
from falcon_challenge.evaluator import FalconEvaluator, DATASET_HELDINOUT_MAP

from context_general_bci.falcon_decoder import NDT2Decoder
# from scripts.falcon_local import run_evaluate

pl.seed_everything(0)

# argparse the eval set
import sys
import argparse

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    # VARIANT = 'h1_single'
    # VARIANT = 'm2_single'
    VARIANT = 'm1_single'
else:
    # This indicates the code is likely running in a shell or other non-Jupyter environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", "-v", type=str, required=True, choices=[
            'h1_single', 
            'h1_joint', 
            'm1_single',
            'm1_joint', 
            'm2_single',
            'm2_joint',
        ]
    )
    args = parser.parse_args()
    VARIANT = args.variant
eval_set = VARIANT.split('_')[0]

eval_paths = Path('./data/falcon_metrics')
eval_paths.mkdir(exist_ok=True, parents=True)
eval_metrics_path = eval_paths / f"{eval_set}_eval_ndt2.csv"
ndt2_run_df = pd.read_csv(eval_metrics_path) if eval_metrics_path.exists() else pd.DataFrame()

#%%
ndt2_run_df.head()
ndt2_run_df = ndt2_run_df[~ndt2_run_df.variant.isin(['h1_oracle_joint-sweep-simple_scratch'])]
def reduce_dataset_from_variant(variant):
    if 'm2' in variant:
        if 'ses' in variant:
            return variant.split('ses-')[-1].split('.*')[0].replace('-', '')
        else:
            return variant.split('.*_')[1].split('_')[0]
    elif 'h1' in variant:
        return variant.split('FALCONH1-')[1].split('_')[0]
    elif 'm1' in variant:
        if 'held_out_oracle' in variant:
            return variant.split('_held_out_oracle-')[0].split('FALCONM1-L_')[-1]
        else:
            return variant.split('ses-')[-1].split('.*')[0].replace('-', '')
    else:
        raise ValueError(f"Unknown variant {variant}")
def get_own_day_r2(row):
    # datasets are reduced, so we can't use falcon-challenge directly for h1/m2.
    if 'h1' in row.variant:
        if row.dataset in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']:
            return row[f'Held In {row.dataset} R2']
        elif row.dataset in ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']:
            return row[f'Held Out {row.dataset} R2']
    elif 'm2' in row.variant:
        if row.dataset in ['20201019', '20201020', '20201027', '20201028']:
            return row[f'Held In {row.dataset} R2']
        elif row.dataset in ['20201030', '20201118', '20201119', '20201124']:
            return row[f'Held Out {row.dataset} R2']
    elif 'm1' in row.variant:
        if row.dataset in DATASET_HELDINOUT_MAP['m1']['held_in']:
            return row[f'Held In {row.dataset} R2']
        elif row.dataset in DATASET_HELDINOUT_MAP['m1']['held_out']:
            return row[f'Held Out {row.dataset} R2']
    print(row)
    raise ValueError(f"Unknown dataset {row.dataset}")

ndt2_run_df['dataset'] = ndt2_run_df.variant.apply(reduce_dataset_from_variant)
ndt2_run_df['Heldout_Accuracy'] = ndt2_run_df.apply(get_own_day_r2, axis=1)

def grouped_heldout_accuracy(df):
    if 'h1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'])]['Heldout_Accuracy']
    elif 'm2' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['20201030', '20201118', '20201119', '20201124'])]['Heldout_Accuracy']
    elif 'm1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(DATASET_HELDINOUT_MAP['m1']['held_out'])]['Heldout_Accuracy']
    return subset.mean(), subset.std()

def grouped_heldin_accuracy(df):
    if 'h1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])]['Heldout_Accuracy']
    elif 'm2' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['20201019', '20201020', '20201027', '20201028'])]['Heldout_Accuracy']
    elif 'm1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(DATASET_HELDINOUT_MAP['m1']['held_in'])]['Heldout_Accuracy']
    return subset.mean(), subset.std()

print(ndt2_run_df[['dataset', 'Heldout_Accuracy']])

print(grouped_heldout_accuracy(ndt2_run_df))
print(grouped_heldin_accuracy(ndt2_run_df))

def compute_other_held_in_mean_perf(df):
    if 'h1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])]
        subset['Other_Accuracy'] = subset.apply(lambda row: sum(
            [row[f'Held In {held_in} R2'] for held_in in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'] if held_in != row.dataset]) / 5, axis=1)
    elif 'm2' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(['20201019', '20201020', '20201027', '20201028'])]
        subset['Other_Accuracy'] = subset.apply(lambda row: sum(
            [row[f'Held In {held_in} R2'] for held_in in ['20201019', '20201020', '20201027', '20201028'] if held_in != row.dataset]) / 3, axis=1)
    elif 'm1' in df.variant.iloc[0]:
        subset = df[df.dataset.isin(DATASET_HELDINOUT_MAP['m1']['held_in'])]
        subset['Other_Accuracy'] = subset.apply(lambda row: sum(
            [row[f'Held In {held_in} R2'] for held_in in DATASET_HELDINOUT_MAP['m1']['held_in'] if held_in != row.dataset]) / (len(DATASET_HELDINOUT_MAP['m1']['held_in']) - 1), axis=1)
    else:
        raise ValueError(f"Unknown variant {df.variant.iloc[0]}")
    # find the best row
    best_held_in_day = subset.loc[subset.groupby('eval_set')['Other_Accuracy'].idxmax()]
    return best_held_in_day
    # return best_hexld_in_day

static_row = compute_other_held_in_mean_perf(ndt2_run_df)
print(f"Static")
print(static_row['heldin_eval_r2'])
print(static_row['eval_r2'])
print(static_row['Held Out R2 Std.'])