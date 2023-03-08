#%%
import numpy as np
import pandas as pd
import h5py
import torch

import logging

from contexts import context_registry
from config import DatasetConfig, DataKey, MetaKey
from config.presets import FlatDataConfig
from data import SpikingDataset
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from analyze_utils import prep_plt
from scipy.io import loadmat

# Clean data from the wilds of the lab

# The goal here is to check for wildly inappropriate spike count distributions.
# We want to plot this _per session_

# Actually easier to do this on the raw data? Presumably? Where things are concatenated?

# dataset_name = 'unstructured_CRS02bLab_session_5_set_4_type_free_play'
contexts = context_registry.query(task=ExperimentalTask.unstructured)
def get_context_spikes(context):
    datapath = context.datapath
    payload = loadmat(datapath, simplify_cells=True, variable_names=['thin_data'])
    return payload['thin_data']['SpikeCount']

df = {'session': [], 'channel': [], 'fr': []}
for context in contexts:
    spikes = get_context_spikes(context) # Time x Channel
    # Find the "anomalous" counts - and suppress them
    elements = np.prod(spikes.shape)
    unique, counts = np.unique(spikes, return_counts=True)
    for u, c in zip(unique, counts):
        if u >= 15 or c / elements < 1e-5: # anomalous, suppress. (Some bins randomly report impossibly high counts like 90 (in 20ms))
            spikes[spikes == u] = 0

    # Too much data...
    # df['session'].extend([context.session] * spikes.shape[0] * spikes.shape[1])
    # df['channel'].extend(np.arange(spikes.shape[1]).repeat(spikes.shape[0]))
    # df['fr'].extend(spikes.flatten())

    # Get some summaries for things that would fail the model
    # It's infeasible to actually manually inspect most of these, so I guess the smart, careful way is to add loss tracking per session and examine for model instability. Not exactly infra I can prioritize atm.
    avg_firing = spikes.mean(axis=0)
    # avg_firing = spikes.std(axis=0)
    # avg_firing = spikes.max(axis=0)

    # # Create histogram of spikes
    df['session'].extend([context.session] * len(avg_firing))
    df['channel'].extend(range(len(avg_firing)))
    df['fr'].extend(avg_firing)
df = pd.DataFrame(df)
#%%
# restrict to first 10 sessions
# sessions = df['session'].unique()
# trunc_df = df[df['session'].isin(sessions[:10])]

# Plot histograms all on the same plot
sns.histplot(x='fr', data=df, hue='session', multiple='layer', legend=False)

# pal = sns.cubehelix_palette(len(trunc_df['session'].unique()), rot=-.25, light=.7)
# g = sns.FacetGrid(trunc_df, row="session", hue="session", aspect=1, height=5, palette=pal)
# g.map(sns.histplot, "fr", element="step", fill=False)
#%%
# sns.histplot(x='fr', data=df, log_scale=True, )
#%%
print(df.groupby('session').fr.max())
sns.histplot(df.groupby('session').fr.max())
# print(len(df['session'].unique()))

#%%