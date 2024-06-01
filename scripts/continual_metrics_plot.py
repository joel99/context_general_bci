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

from falcon_challenge.evaluator import FalconEvaluator, DATASET_HELDINOUT_MAP

pl.seed_everything(0)

# argparse the eval set
import sys
import argparse

num_workers = 4 # for main eval block.
if 'ipykernel' in sys.modules:
    print("Running in a Jupyter notebook.")
    # VARIANT = 'h1'
    VARIANT = 'm2'
    # VARIANT = 'm1'
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
    args = parser.parse_args()
    VARIANT = args.variant
eval_set = VARIANT.split('_')[0]

eval_paths = Path('./data/falcon_metrics')
eval_paths.mkdir(exist_ok=True, parents=True)
eval_metrics_path_cont = eval_paths / f"{eval_set}_continual_ndt2.csv"
eval_metrics_path_tr = eval_paths / f"{eval_set}_trialized_ndt2.csv"
ndt2_run_df_cont = pd.read_csv(eval_metrics_path_cont) if eval_metrics_path_cont.exists() else pd.DataFrame()
ndt2_run_df_tr = pd.read_csv(eval_metrics_path_tr) if eval_metrics_path_tr.exists() else pd.DataFrame()
ndt2_run_df_cont['type'] = 'continual'
ndt2_run_df_tr['type'] = 'trialized'
ndt2_run_df = pd.concat([ndt2_run_df_cont, ndt2_run_df_tr])
print(ndt2_run_df)
#%%
from context_general_bci.analyze_utils import prep_plt
ax = prep_plt()
ndt2_run_df.head()
ndt2_run_df['perf_diff'] = ndt2_run_df['heldin_eval_r2'] - ndt2_run_df['eval_r2']
ax = sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='eval_r2', hue='type', ax=ax)
ax = sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='heldin_eval_r2', hue='type', ax=ax)
# sns.lineplot(data=ndt2_run_df, x='augment_chop_length_ms', y='eval_r2', hue='type')
ax.set_ylim(0, 1)

