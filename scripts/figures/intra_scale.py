#%%
# Probably the main point of this minor plot is to demonstrate that there _is_ scaling; it's almost impossible to tell whether the trend is slowing at this point.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger

from analyze_utils import stack_batch, load_wandb_run
from analyze_utils import prep_plt, get_dataloader
from utils import wandb_query_experiment, get_wandb_run, wandb_query_latest

pl.seed_everything(0)

UNSORT = True
UNSORT = False

EXPERIMENTS_NLL = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}',
]

queries = [
    's100',
    's200',
    's400',
    's800',
    's1600',
    's3200',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_nll = wandb_query_experiment(EXPERIMENTS_NLL, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed', 'running']},
})

#%%
def get_evals(model, dataloader, runs=8, mode='nll'):
    evals = []
    for i in range(runs):
        pl.seed_everything(i)
        heldin_metrics = stack_batch(trainer.test(model, dataloader, verbose=False))
        if mode == 'nll':
            test = heldin_metrics['test_infill_loss'] if 'test_infill_loss' in heldin_metrics else heldin_metrics['test_shuffle_infill_loss']
        else:
            test = heldin_metrics['test_kinematic_r2']
        test = test.mean().item()
        evals.append({
            'seed': i,
            mode: test,
        })
    return pd.DataFrame(evals)[mode].mean()
    # return evals

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        if 'frag' not in run.name:
            continue
        variant, _frag, *rest = run.name.split('-')
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        dataset_name = cfg.dataset.datasets[0] # drop wandb ID
        if (variant, dataset_name, run.config['model']['lr_init']) in seen_set:
            continue
        dataset = SpikingDataset(cfg.dataset)
        dataset.subset_split(splits=['eval'])
        dataset.build_context_index()
        data_attrs = dataset.get_data_attrs()
        model = transfer_model(src_model, cfg.model, data_attrs)
        dataloader = get_dataloader(dataset)
        payload = {
            'limit': run.config['dataset']['scale_limit_per_eval_session'],
            'variant': variant,
            'dataset': dataset_name,
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init'])] = True
    return pd.DataFrame(df)
df = build_df(runs_nll, mode='nll')

#%%
# Show just NLL in logscale
palette = sns.color_palette('colorblind', n_colors=len(df['dataset'].unique()))
dataset_order = df.groupby(['dataset']).mean().sort_values('nll').index
ax = prep_plt()
ax = sns.scatterplot(
    x='limit',
    y='nll',
    hue='dataset',
    hue_order=dataset_order,
    data=df,
    palette=palette,
    ax=ax,
    legend=False
)
ax.set_xscale('log')
ax.set_yscale('log')

# Fit power law to the minimum test loss vs. training steps
from scipy.optimize import curve_fit
def power_law(x, a, b):
    return a * x**b

def plot_dataset_power_law(df, dataset, ax, **kwargs):
    sub_df = df[df['dataset'] == dataset]
    popt, pcov = curve_fit(power_law, sub_df['limit'], sub_df['nll'])
    x = torch.linspace(sub_df['limit'].min(), sub_df['limit'].max(), 100)
    y = power_law(x, *popt)
    ax.plot(x, y, linestyle='--', **kwargs)
    # annotate with power law
    ax.annotate(f'{popt[1]:.4f}', xy=(x[0], y[0]), xytext=(x[0] + 10, y[0]), **kwargs)


for i, dataset in enumerate(dataset_order):
    plot_dataset_power_law(df, dataset, ax, color=palette[i])

ax.set_title(f'Intra-session scaling ({"unsorted" if UNSORT else "sorted"})')