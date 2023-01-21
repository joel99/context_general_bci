#%%
r"""
    TODO
    Identify .mat files for relevant trials
    Open .mat
    Find the phases (observation) that are relevant for now
    Find the spikes
    Find the observed kinematic traces

    Batch for other sessions
"""
import pandas as pd
import numpy as np
# import xarray as xr
from pathlib import Path
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch

data_dir = Path("./data/pitt_co/mat/")
session = 329

session_dir = data_dir / f'CRS02bHome.data.{session:05d}'

if not session_dir.exists():
    print(f'Session {session_dir} not found; Run `prep_all` on the QL .bin files.')

def extract_ql_data(ql_data):
    # ql_data: .mat['iData']['QL']['Data']
    # Currently just equipped to extract spike snippets
    # If you want more, look at `icms_modeling/scripts/preprocess_mat`
    # print(ql_data.keys())
    # print(ql_data['TASK_STATE_CONFIG'].keys())
    # print(ql_data['TASK_STATE_CONFIG']['state_num'])
    # print(ql_data['TASK_STATE_CONFIG']['state_name'])
    # print(ql_data['TRIAL_METADATA'])
    def extract_spike_snippets(spike_snippets):
        THRESHOLD_SAMPLE = 12./30000
        return {
            "spikes_source_index": spike_snippets['source_index'], # JY: I think this is NSP box?
            "spikes_channel": spike_snippets['channel'],
            "spikes_source_timestamp": spike_snippets['source_timestamp'] + THRESHOLD_SAMPLE,
            # "spikes_snippets": spike_snippets['snippet'], # for waveform
        }

    return {
        **extract_spike_snippets(ql_data['SPIKE_SNIPPET']['ss'])
    }

def events_to_raster(
    events,
    channels_per_array=128,
):
    """
        Tensorize sparse format.
    """
    events['spikes_channel'] = events['spikes_channel'] + events['spikes_source_index'] * channels_per_array
    bins = np.arange(
        events['spikes_source_timestamp'].min(),
        events['spikes_source_timestamp'].max(),
        0.001
    )
    timebins = np.digitize(events['spikes_source_timestamp'], bins, right=False) - 1
    spikes = torch.zeros((len(bins), 256), dtype=torch.uint8)
    spikes[timebins, events['spikes_channel']] = 1
    return spikes


def load_trial(fn, use_ql=True):
    # if `use_ql`, use the prebinned at 20ms and also pull out the kinematics
    # else take raw spikes
    payload = scipy.io.loadmat(fn, simplify_cells=True)
    # data = payload['data'] # 'data' is pre-binned at 20ms, we'd rather have more raw
    # payload = scipy.io.loadmat(fn, simplify_cells=True, variable_names=['iData'])
    # print(payload['data']['TaskStateMasks']['states'])
    # print(payload['data']['TaskStateMasks']['state_num'])
    out = {
        'bin_size_ms': 20 if use_ql else 1,
        'use_ql': use_ql,
    }
    if use_ql:
        # print(payload['data'].keys())
        # print(payload['data']['SpikeCount'].shape)
        # print(payload['data']['ActiveChannelMask'].sum())
        standard_channels = np.arange(0, 256 * 5,5) # unsorted, I guess
        spikes = payload['data']['SpikeCount'][..., standard_channels]
        # print(payload['data']['Kinematics'].keys())
        out['spikes'] = torch.from_numpy(spikes)
        # cursor x, y
        out['position'] = torch.from_numpy(payload['data']['Kinematics']['ActualPos'][:,2:4])
        print(payload['data'].keys())
    else:
        data = payload['iData']
        trial_data = extract_ql_data(data['QL']['Data'])
        out['src_file'] = data['QL']['FileName']
        out['spikes'] = events_to_raster(trial_data)
    return out

for fname in session_dir.glob("*.mat"):
    if fname.stem.startswith('QL.Task'):
        payload = load_trial(fname)
        break

#%%
print(payload['spikes'].shape)

#%%
from contexts import context_registry
# print(context_registry.search_index.id)
print(context_registry.query(alias='CRS02bHome.data.00329'))
# print('')

#%%
# Make raster plot
fig, ax = plt.subplots(figsize=(10, 10))
from analyze_utils import prep_plt

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
    # time_lim = spikes.shape[0] * 0.001
    # ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    # ax.set_xticklabels(np.linspace(0, time_lim, 5))
    # ax.set_title("Benchmark Maze (Sorted)")
    # ax.set_title(context.alias)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax
plot_spikes(payload['spikes'])


#%%