#%%
# Corresponding to exp/arch
import logging
import sys
from typing import List
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import seaborn as sns
import torch
import pytorch_lightning as pl
from einops import rearrange

from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs

from analyze_utils import get_run_config, load_wandb_run, wandb_query_latest, wandb_query_experiment
from analyze_utils import prep_plt

from matplotlib.colors import LogNorm, Normalize

def get_run_dict(run):
    keys = [
        'trainer/global_step',
        'eval_loss',
        'val_loss',
        'epoch',
    ]
    df = run.history(
        samples=1000, # unfortunately heavy since we need log scale
        keys=keys
    )
    out = {}
    out['val_loss'] = [df.loc[df['val_loss'].idxmin()]['val_loss']]
    out['test_loss'] = [df.loc[df['val_loss'].idxmin()]['eval_loss']]
    out['id'] = [run.id]
    out['tag'] = [run.config['tag']]
    out['experiment'] = [run.config['experiment_set']]
    out = pd.DataFrame(out)
    return out

def get_run_df(runs, labels):
    return pd.concat([
        get_run_dict(run) for run in runs
    ])


#%%
title = 'Sweep'

# pull experiment
experiment = [
    'arch/tune_hp',
]

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished']},
    duration={"$gt": 300},
)

run_labels = [
    'single_f8-sweep_base_v2'
]


df = get_run_df(runs, run_labels)
#%%
crop_labels = lambda x: x.split('-')[0]
df['label'] = df['tag'].apply(crop_labels)
ax = prep_plt()
sns.boxplot(
    data=df,
    x='label',
    y='test_loss',
    # hue='variant',
    ax=ax,
)
