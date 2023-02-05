#%%
import numpy as np
import pandas as pd
import h5py
import torch

import logging

from contexts import context_registry
from config import DatasetConfig, DataKey, MetaKey
from data import SpikingDataset
from tasks import ExperimentalTask

from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from analyze_utils import prep_plt, wandb_query_latest, load_wandb_run

mode = 'rtt'
mode = 'pitt'
if mode == 'rtt':
    ctxs = context_registry.query(task=ExperimentalTask.odoherty_rtt)
else:
    ctxs = context_registry.query(task=ExperimentalTask.observation)

context = ctxs[0]
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

sample_query = mode # just pull the latest run
wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
default_cfg = cfg.dataset
# default_cfg: DatasetConfig = OmegaConf.create(DatasetConfig())
# default_cfg.data_keys = [DataKey.spikes]
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
# default_cfg.bin_size_ms = 5
# default_cfg.bin_size_ms = 20
# print(default_cfg.datasets )
# default_cfg.datasets = [context.alias]
default_cfg.max_arrays = min(max(1, len(context.array)), 2)
# default_cfg.max_channels = 250
dataset = SpikingDataset(default_cfg)
dataset.build_context_index()

# import torch
# lengths = []
# for t in range(1000):
#     lengths.append(dataset[t][DataKey.spikes].size(0))
# print(torch.tensor(lengths).max(), torch.tensor(lengths).min())
print(len(dataset))
#%%
trial = 0
# trial = 10
trial_vel = dataset[trial][DataKey.bhvr_vel]

# Show kinematic trace by integrating trial_vel
print(trial_vel.shape)
trial_pos = trial_vel.cumsum(0)
trial_pos = trial_pos - trial_pos[0]
# # Plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(trial_vel)
ax[0].set_title('Velocity')
ax[1].plot(trial_pos)
ax[1].set_title('Position')

#%%
trial = 10
trial = 26

pop_spikes = dataset[trial][DataKey.spikes]
pop_spikes = pop_spikes[..., 0]
# print diagnostics
# print(pop_spikes[::2].sum(0))
# print(pop_spikes[1::2].sum(0))
# sns.histplot(pop_spikes[::2].sum(0))
# sns.histplot(pop_spikes[1::2].sum(0) - pop_spikes[0::2].sum(0))
print(
    f"Mean: {pop_spikes.float().mean():.2f}, \n"
    f"Std: {pop_spikes.float().std():.2f}, \n"
    f"Max: {pop_spikes.max():.2f}, \n"
    f"Min: {pop_spikes.min():.2f}, \n"
    f"Shape: {pop_spikes.shape}"
)

pop_spikes = pop_spikes.flatten(1, 2)
# pop_spikes = pop_spikes[:, :96]
# wait... 250?
# path_to_old = './data/old_nlb/mc_maze.h5'
# with h5py.File(path_to_old, 'r') as f:
#     print(f.keys())
#     pop_spikes = f['train_data_heldin']
#     pop_spikes = torch.tensor(pop_spikes)
#     print(pop_spikes.shape)
# pop_spikes = pop_spikes[trial]

print(pop_spikes.shape)
# print(pop_spikes.sum(0) / 0.6)
# print(pop_spikes.sum(0))
# Build raster scatter plot of pop_spikes
def plot_spikes(spikes, ax=None, vert_space=1):

    if ax is None:
        fig, ax = plt.subplots()
    ax = prep_plt(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    spike_t, spike_c = np.where(spikes)
    # prep_plt(axes[_c], big=True)
    time = np.arange(spikes.shape[0])
    ax.scatter(
        time[spike_t], spike_c * vert_space,
        # c=colors,
        marker='|',
        s=10,
        alpha=0.9
        # alpha=0.3
    )
    time_lim = spikes.shape[0] * dataset.cfg.bin_size_ms
    ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    ax.set_xticklabels(np.linspace(0, time_lim, 5))
    # ax.set_title("Benchmark Maze (Sorted)")
    ax.set_title(context.alias)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax
plot_spikes(pop_spikes)
