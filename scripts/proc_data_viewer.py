#%%
import numpy as np
import pandas as pd
import h5py
import torch

import logging

from contexts import context_registry
from config import DatasetConfig, DataKey, MetaKey
from data import SpikingDataset

from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from utils import prep_plt

# dataset_name = 'mc_maze_large' # 122 sorted units
# dataset_name = 'mc_maze_medium' # 114 sorted units
dataset_name = 'mc_maze_small' # 107 sorted units
# dataset_name = 'mc_maze$' # 137 sorted units
# dataset_name = 'churchland_maze_jenkins-0'
# dataset_name = 'churchland_maze_jenkins-1'
# dataset_name = 'churchland_maze_jenkins-2'
# dataset_name = 'churchland_maze_jenkins-3'
# dataset_name = 'churchland_maze_nitschke-0'
# dataset_name = 'churchland_maze_nitschke-1'
# dataset_name = 'churchland_maze_nitschke-2'
# dataset_name = 'churchland_maze_nitschke-3'
# dataset_name = 'churchland_maze_nitschke-4'

# TODO - current inferred rates for RTT are wavy for some reason
# dataset_name = 'mc_rtt'
# dataset_name = 'odoherty_rtt-Loco-20170215_02'

# dataset_name = 'odoherty_rtt-Loco-20170216_02'
# dataset_name = 'odoherty_rtt-Loco-20170217_02'
# dataset_name = 'odoherty_rtt-Loco-20170228_02'
# dataset_name = 'odoherty_rtt-Loco-20170301_05'
# dataset_name = 'odoherty_rtt-Loco-20170302_02'
# dataset_name = 'odoherty_rtt-Loco-20170227_04'
# dataset_name = 'odoherty_rtt-Loco-20170210_03'

# dataset_name = 'odoherty_rtt-Indy.*'
# dataset_name = 'odoherty_rtt-Indy-20160627_01'
# dataset_name = 'odoherty_rtt-Indy-20161005_06'
# dataset_name = 'odoherty_rtt-Indy-20160630_01'
# dataset_name = 'odoherty_rtt-Indy-20160915_01'
# dataset_name = 'odoherty_rtt-Indy-20160916_01'
# dataset_name = 'odoherty_rtt-Indy-20160921_01'
context = context_registry.query(alias=dataset_name)
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

default_cfg: DatasetConfig = OmegaConf.create(DatasetConfig())
default_cfg.data_keys = [DataKey.spikes]
# default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.bin_size_ms = 5
default_cfg.max_arrays = min(max(1, len(context.array)), 2)
default_cfg.datasets = [context.alias]
dataset = SpikingDataset(default_cfg)
dataset.build_context_index()

# import torch
# lengths = []
# for t in range(1000):
#     lengths.append(dataset[t][DataKey.spikes].size(0))
# print(torch.tensor(lengths).max(), torch.tensor(lengths).min())
print(len(dataset))
# #%%
# trial = 0
# trial = 10
# trial_vel = dataset[trial][DataKey.bhvr_vel]

# # Show kinematic trace by integrating trial_vel
# trial_pos = trial_vel.cumsum(0)
# trial_pos = trial_pos - trial_pos[0]
# # Plot
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(trial_vel)
# ax[1].plot(trial_pos)

#%%
# trial = 10
trial = 26

pop_spikes = dataset[trial][DataKey.spikes]
pop_spikes = pop_spikes[..., 0]
# print diagnostics
print(pop_spikes[::2].sum(0))
print(pop_spikes[1::2].sum(0))
# sns.histplot(pop_spikes[::2].sum(0))
# sns.histplot(pop_spikes[1::2].sum(0) - pop_spikes[0::2].sum(0))
print(
    f"Mean: {pop_spikes.float().mean():.2f}, "
    f"Std: {pop_spikes.float().std():.2f}, "
    f"Max: {pop_spikes.max():.2f}, "
    f"Min: {pop_spikes.min():.2f}"
    f"Shape: {pop_spikes.shape}"
)

pop_spikes = pop_spikes.flatten(1, 2)
pop_spikes = pop_spikes[:, :96]

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
