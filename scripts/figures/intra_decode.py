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
# UNSORT = False

DATASET_WHITELIST = [
    "odoherty_rtt-Indy-20160407_02",
    "odoherty_rtt-Indy-20170131_02",
    "odoherty_rtt-Indy-20160627_01",
]

EXPERIMENTS_KIN = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/probe',
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/decode',
    f'scale_decode/probe',
    # f'scale_decode/probe/mix',
]

queries = [
    's100',
    's200',
    's400',
    's800',
    's1600',
    's3200',
    'sup_20',
    'sup_100',
    'sup_200',
    'sup_800',
    'sup_3200',
    'unsup_20',
    'unsup_100',
    'unsup_200',
    'unsup_800',
    'unsup_3200',
]

merge_queries = [
    f'{q}-frag-{d}' for q in queries for d in DATASET_WHITELIST
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
    "config.dataset.odoherty_rtt.include_sorted": not UNSORT,
})
runs_kin = [r for r in runs_kin if r.config['dataset']['datasets'][0] in DATASET_WHITELIST and r.name.split('-')[0] in queries]

print(f'Found {len(runs_kin)} runs')
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
        if dataset_name not in DATASET_WHITELIST:
            continue
        experiment_set = run.config['experiment_set']
        if variant.startswith('sup') or variant.startswith('unsup'):
            experiment_set = experiment_set + '_' + variant.split('_')[0]
        if (
            variant,
            dataset_name,
            run.config['model']['lr_init'],
            experiment_set
        ) in seen_set:
            continue
        dataset = SpikingDataset(cfg.dataset)
        set_limit = run.config['dataset']['scale_limit_per_eval_session']
        # if set_limit == 0:
            # train_dev_dataset = SpikingDataset(cfg.dataset)
            # train_dev_dataset.subset_split()
            # set_limit = len(train_dev_dataset)
        dataset.subset_split(splits=['eval'])
        dataset.build_context_index()
        data_attrs = dataset.get_data_attrs()
        model = transfer_model(src_model, cfg.model, data_attrs)
        dataloader = get_dataloader(dataset)

        payload = {
            'limit': set_limit,
            'variant': variant,
            'series': experiment_set,
            'dataset': dataset_name,
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init']), experiment_set] = True
    return pd.DataFrame(df)

kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset', 'series'])

df = kin_df

#%%
print(df[df['series'] == 'scale_v3/intra_unsort/'])
#%%
prescribed_limits = {
    's3200': 3190,
    'unsup_3200': 3190,
    'unsup_800': 770,
    'unsup_200': 200,
    'unsup_100': 100,
    'unsup_20': 20,
    'sup_3200': 3190,
    'sup_800': 770,
    'sup_200': 200,
    'sup_100': 100,
    'sup_20': 20,
    's1600': 1600,
    's800': 770, # relevant for the 2 limited datasets
    's400': 400,
    's200': 200,
    's100': 100,
}
# override `limit` with `prescribed_limits` based on `variant` for `scale_v3/intra_unsort/probe` series
df.loc[df['variant'].isin(prescribed_limits.keys()) & (df['series'] == 'scale_v3/intra_unsort/probe'), 'limit'] = df.loc[df['variant'].isin(prescribed_limits.keys()) & (df['series'] == 'scale_v3/intra_unsort/probe'), 'variant'].map(prescribed_limits)
#%%
palette = sns.color_palette('colorblind', n_colors=len(df['series'].unique()))
hue_order = df['series'].unique()

g = sns.relplot(
    x='limit',
    y='kin_r2',
    style='series',
    hue='series',
    hue_order=hue_order,
    data=df,
    palette=palette,
    # kind='scatter',
    markers=True,
    kind='line',
    facet_kws={'sharex': False, 'sharey': False},
    col='dataset',
    # row='dataset',
)
def deco(data, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_xscale('log')
    ax.set_xlabel('Target session trials')
    ax.set_ylabel('Vel R2')
    # ax.set_yscale('log')

relabel = {
    'scale_v3/intra_unsort/probe': 'Scratch (100 Trial Sup)',
    'scale_v3/intra_unsort/decode': 'Scratch',
    'scale_decode/probe_sup': 'Sup tune',
    'scale_decode/probe_unsup': 'Unsup tune',
}
g._legend.set_title('Variant')
for t, l in zip(g._legend.texts, hue_order):
    t.set_text(relabel[l])


g.map_dataframe(deco)
g.fig.suptitle(f'Tuning a Decoder ({"Unsorted" if UNSORT else "Sorted"})', y=1.05, fontsize=28)
