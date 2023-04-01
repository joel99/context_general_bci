#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
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
EXPERIMENTS = [
    'arch/robust',
    'arch/robust/tune'
]

queries = [
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
runs = wandb_query_experiment(EXPERIMENTS, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
})
#%%
def get_evals(model, dataloader, runs=8):
    evals = []
    for i in range(runs):
        pl.seed_everything(i)
        heldin_metrics = stack_batch(trainer.test(model, dataloader, verbose=False))
        test_nll = heldin_metrics['test_infill_loss'] if 'test_infill_loss' in heldin_metrics else heldin_metrics['test_shuffle_infill_loss']
        test_nll = test_nll.mean().item()
        evals.append({
            'seed': i,
            'test_nll': test_nll,
        })
    return pd.DataFrame(evals)['test_nll'].mean()
    # return evals

df = []
seen_set = {}
for run in runs:
    if 'frag' not in run.name:
        continue
    variant, _frag, *rest = run.name.split('-')
    dataset_name = '-'.join(rest[:-1]) # drop wandb ID
    if (variant, dataset_name) in seen_set:
        continue
    src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    model = transfer_model(src_model, cfg.model, data_attrs)
    dataloader = get_dataloader(dataset)
    payload = {
        'variant': variant,
        'dataset': dataset_name,
        'test_nll': get_evals(model, dataloader)
    }
    df.append(payload)
    seen_set[(variant, dataset_name)] = True
df = pd.DataFrame(df)
#%%
# get unique counts - are all the runs done?
# print(df)
# df.groupby(['variant']).count()

#%%
# Show just NLL
ax = sns.barplot(
    # x='dataset',
    # hue='variant',
    x='variant',
    y='test_nll',
    data=df
)
# rotate x labels
for item in ax.get_xticklabels():
    item.set_rotation(45)

#%%
# make facet grid with model cali
g = sns.FacetGrid(
    df,
    col='dataset',
    col_wrap=3,
    sharey=False,
    sharex=False,
    height=3,
    aspect=1.5,
)


# # plot test nll
g.map_dataframe(
    # sns.stripplot,
    sns.scatterplot,
    x='test_nll',
    y='test_nll',
    hue='variant',
    # dodge=True,
)

# add legend
g.add_legend()
# Set xlabel to calibation trials
# g.set_axis_labels('Calibration trials', 'Test NLL')

# ax = prep_plt()

# ax = sns.scatterplot(
# ax = sns.stripplot(
    # x='dataset',
    # x='model_cali',
    # hue='model_type',
    # y='test_nll',
    # dodge=True,
    # data=all_evals,
    # ax=ax
# )
# ax.set_title('Test NLL across eval seeds')

# rotate xlabels
# for item in ax.get_xticklabels():
    # item.set_rotation(45)
