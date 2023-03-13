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

sample_query = 'test' # just pull the latest run
sample_query = 'pt_parity'

wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
cfg.dataset.datasets = ['observation_.*']
# default_cfg: DatasetConfig = OmegaConf.create(DatasetConfig())
# default_cfg.data_keys = [DataKey.spikes]
cfg.dataset.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
dataset = SpikingDataset(cfg.dataset)
dataset.build_context_index()
dataset.subset_split()

# import torch
# lengths = []
# for t in range(1000):
#     lengths.append(dataset[t][DataKey.spikes].size(0))
# print(torch.tensor(lengths).max(), torch.tensor(lengths).min())
print(len(dataset))
#%%
vels = []
for t in range(len(dataset)):
    vels.append(dataset[t][DataKey.bhvr_vel])
vels = torch.cat(vels, 0)
print(vels.shape)
#%%
torch.save({
    'mean': vels.mean(0),
    'std': vels.std(0),
}, 'pitt_obs_zscore.pt')
# print(vels.mean(0), vels.std(0))
# print(vels.min(0), vels.max(0))
# print((vels / vels.std(0)).min(0), (vels / vels.std(0)).max(0))

#%%
# trial = 0
trial = 10
# trial = 30
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
# iterate through trials and print min and max bhvr_vel
min_vel = 0
max_vel = 0
for trial in range(len(dataset)):
    trial_vel = dataset[trial][DataKey.bhvr_vel]
    min_vel = min(min_vel, trial_vel.min())
    max_vel = max(max_vel, trial_vel.max())
print(min_vel, max_vel)

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
