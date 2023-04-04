#%%
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

PLOT_DECODE = False
USE_SORTED = True
# USE_SORTED = False
exp_tag = f'robust{"" if USE_SORTED else "_unsort"}'
EXPERIMENTS_KIN = [
    f'arch/{exp_tag}/probe',
]
EXPERIMENTS_NLL = [
    f'arch/{exp_tag}',
    f'arch/{exp_tag}/tune',
]

queries = [
    'single_f8',
    'f8',
    'subject_f8',
    'task_f8',
    'single_time',
    'single_f32',
    'f32',
    'stitch',
    'subject_f32',
    'subject_stitch',
    'task_f32',
    'task_stitch',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
    "state": {"$in": ['finished']},
})
runs_nll = wandb_query_experiment(EXPERIMENTS_NLL, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
    "state": {"$in": ['finished']},
})
# %%
runs_kin = [r for r in runs_kin if r.name.split('-')[0] in queries]
runs_nll = [r for r in runs_nll if r.name.split('-')[0] in queries]
print([r.name for r in runs_nll])
print(len(runs_nll))
print(len(runs_kin)) # 4 * 5 * 3
# runs_kin = runs_kin[:10]
# runs_nll = runs_nll[:10]
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
        dataloader = get_dataloader(dataset, batch_size=16 if cfg.model.neurons_per_token == 8 else 100)
        payload = {
            'variant': variant,
            'dataset': dataset_name,
            'chunk': run.config['model']['neurons_per_token'],
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init'])] = True
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset'])

nll_df = build_df(runs_nll, mode='nll')

# merge on variant and dataset, filling empty with 0s
df = pd.merge(kin_df, nll_df, on=['variant', 'dataset'], how='outer').fillna(0)

#%%
# Meta-annotation

source_map = {
    'single_f8': 'single',
    'f8': 'session',
    'subject_f8': 'subject',
    'task_f8': 'task',
    'task_f32': 'task',
    'task_stitch': 'task',
    'subject_f32': 'subject',
    'subject_stitch': 'subject',
    'f32': 'session',
    'stitch': 'session',
    'single_f32': 'single',
    'single_time': 'single',
}
arch_map = {
    'single_f8': 'f8',
    'f8': 'f8',
    'subject_f8': 'f8',
    'task_f8': 'f8',
    'task_f32': 'f32',
    'task_stitch': 'stitch',
    'subject_f32': 'f32',
    'subject_stitch': 'stitch',
    'f32': 'f32',
    'stitch': 'stitch',
    'single_f32': 'f32',
    'single_time': 'time',
}
df['source'] = df['variant'].apply(lambda x: source_map[x])
df['arch'] = df['variant'].apply(lambda x: arch_map[x])

# https://docs.google.com/spreadsheets/d/1WpmhgttDJY09IxHzZqfHh5e8vozeI9c42o8xLfibEng/edit?usp=sharing
eRFH_baseline_kin = {
    'odoherty_rtt-Indy-20160407_02': 0.64575,
    'odoherty_rtt-Indy-20160627_01': 0.53125,
    'odoherty_rtt-Indy-20161005_06': 0.484,
    'odoherty_rtt-Indy-20161026_03': 0.5955,
    'odoherty_rtt-Indy-20170131_02': 0.5113,
}

#%%
# print(df)
# print(kin_df.columns)
# get unique counts - are all the runs done?
# print(df)
# df.groupby(['variant']).count()

#%%
# Show just NLL
ax = sns.barplot(
    # x='dataset',
    # hue='variant',
    x='variant',
    y='nll',
    data=df
)
# rotate x labels
for item in ax.get_xticklabels():
    item.set_rotation(45)

#%%
ax = prep_plt()
aggr_variant = df.groupby(['variant', 'source', 'arch']).mean().reset_index()
palette = sns.color_palette('colorblind', len(aggr_variant))
ax = sns.scatterplot(
    x='nll',
    y='kin_r2',
    hue='arch',
    # hue='variant',
    style='source',
    s=100,
    data=aggr_variant,
    palette=palette,
    legend=True,
)

# Annotate the individual datapoints
# for i, row in aggr_variant.iterrows():
#     ax.text(
#         row.nll, row.kin_r2 + 0.005, row.variant, color=palette[i], ha='center', va='bottom',
#         fontsize=14,
#     )

mean_baseline = np.mean([eRFH_baseline_kin[k] for k in eRFH_baseline_kin if k in df.dataset.unique()])
ax.axhline(mean_baseline, ls='--', color='black', label='mean across variants')
# Annotate the horizontal line
ax.text(
    ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, mean_baseline, 'rEFH', color='black', ha='left', va='bottom',
    fontsize=14,
)

#%%
# make facet grid with model cali
sorted_datasets = sorted(df.variant.unique())
palette = sns.color_palette('colorblind', len(aggr_variant))
g = sns.relplot(
    data=df,
    col='dataset',
    x='nll',
    y='kin_r2',
    # hue='variant',
    hue='arch',
    # style='variant',
    style='source',
    s=100,
    col_wrap=3,
    facet_kws={'sharey': False, 'sharex': False}
)
def deco(data, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)

    # Annotate the horizontal line
    mean_baseline = eRFH_baseline_kin[data.dataset.unique()[0]]
    ax.axhline(mean_baseline, ls='--', color='black', label='mean across variants')
    # annotate rEFH position
    ax.text(
        ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, mean_baseline, 'rEFH', color='black', ha='left', va='bottom',
        fontsize=14,
    )
g.map_dataframe(deco)
