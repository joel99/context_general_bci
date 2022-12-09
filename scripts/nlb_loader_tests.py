#%%
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import logging

from contexts import context_registry
from config import DatasetConfig, DataKey, MetaKey
from data import SpikingDataset

from matplotlib import pyplot as plt
import seaborn as sns

# dataset_name = 'mc_maze_large' # 122 sorted units
# dataset_name = 'mc_maze_medium' # 114 sorted units
# dataset_name = 'mc_maze_small' # 107 sorted units
# dataset_name = 'mc_maze' # 137 sorted units
dataset_name = 'churchland_maze_jenkins-0'
dataset_name = 'churchland_maze_jenkins-1'
context = context_registry.query(alias=dataset_name)

default_cfg = DatasetConfig()
default_cfg.bin_size_ms = 5
# default_cfg.max_arrays = 1
default_cfg.max_arrays = 2
dataset = SpikingDataset(default_cfg)

dataset.meta_df = dataset.load_session(dataset_name)[0]
dataset.build_context_index()
# dataset = NWBDataset(context.datapath)

#%%
from utils import prep_plt
trial = 10
pop_spikes = dataset[trial][DataKey.spikes]
pop_spikes = pop_spikes[..., 0]
pop_spikes = pop_spikes.flatten(1, 2)
print(pop_spikes.shape)
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
    ax.set_title(dataset_name)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax
plot_spikes(pop_spikes)
