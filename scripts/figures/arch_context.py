#%%
import logging
import sys
from typing import List
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger

from analyze_utils import get_run_config, load_wandb_run, wandb_query_latest, wandb_query_experiment
from analyze_utils import prep_plt

from matplotlib.colors import LogNorm, Normalize
import wandb

# pull experiment
experiment = ['arch/context']

# Quickly pull a run to get full dataset size
def get_train_dataset_size(sample='scale1_1s'):
    wandb_run = wandb_query_latest(sample, exact=False, allow_running=True)[0]
    cfg = get_run_config(wandb_run, tag='val_loss')
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split()
    train, _ = dataset.create_tv_datasets()
    return len(train)
train_size = get_train_dataset_size()

# call wandb api to pull all runs with this experiment tag

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished', 'failed', 'crashed', 'running']},
    duration={"$gt": 300},
)

def get_run_dict(run):
    out = run.history(
        samples=1000, # unfortunately heavy since we need log scale
        keys=[
            'trainer/global_step',
            'eval_loss',
            'epoch',
        ]
    )
    out['id'] = run.id
    out['tag'] = run.config['tag']
    out['scale_factor'] = run.config['dataset']['scale_ratio']
    if run.config['model']['session_embed_strategy'] == str(EmbedStrat.none):
        out['context_tokens'] = 0
    else:
        out['context_tokens'] = run.config['model']['session_embed_token_count']
    out['token_size'] = run.config['model']['neurons_per_token']
    out['trials'] = round(out['scale_factor'] * train_size)
    out['n_layers'] = run.config['model']['transformer']['n_layers']
    return out

def get_run_df(runs, labels):
    return pd.concat([
        get_run_dict(run) for run in runs if run.config['tag'] in labels
    ])

# run_dicts = [get_run_dict(run) for run in runs]
#%%
title = "Increased context capacity speeds convergence"
run_labels = [
    "scale1_0s", "scale1_1s", "scale1_8s",
    "scale4_0s", "scale4_1s", "scale4_8s",
    "scale16_0s", "scale16_1s", "scale16_8s",
]

df = get_run_df(runs, run_labels)

cmap = sns.color_palette('viridis_r', n_colors=3)
# cmap = sns.color_palette('viridis_r', as_cmap=True)
# Make a facet grid with one row per scale factor
g = sns.FacetGrid(
    data=df,
    row='trials',
    row_order=sorted(df['trials'].unique()),
    sharex=True,
    sharey=True,
    height=3,
    aspect=2,
    gridspec_kws={'hspace': 0.05},
)
g.map_dataframe(
    sns.lineplot,
    x='trainer/global_step',
    y='eval_loss',
    hue='context_tokens',
    style='context_tokens',
    # style_order=reversed(sorted(df['context_tokens'].unique())),
    palette=cmap,
    alpha=0.8,
    legend=True,
    # legend=False,
)

g.add_legend(title='Context Tokens', bbox_to_anchor=(0.8, 0.4), loc='lower right')


labels = sorted(df['trials'].unique())
for i, ax in enumerate(g.axes.flat):
    ax = prep_plt(ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(4e2, ax.get_xlim()[1])
    ax.set_ylim(ax.get_ylim()[0], 0.6)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Test loss')
    ax.set_title('')
    # Annotate with label on top right of plot
    ax.text(
        0.9, 0.6,
        # only use 3 sigfigs
        rf'$S={labels[i]:.2g}$ trials',
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=15,
    )

    # Only mark min and max on y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # rotate ytick labels by 90 degrees
    # do not use scientific notation
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.tick_params(axis='y', rotation=90)



#%%
title = 'Not bottlenecked by model size'
# TODO
