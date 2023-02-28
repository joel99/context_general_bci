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

from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from analyze_utils import prep_plt
# Convert the above to valid python syntax


datasets = [
    'odoherty_rtt-Indy-20160407_02',
    'odoherty_rtt-Indy-20160411_01',
    'odoherty_rtt-Indy-20160411_02',
    'odoherty_rtt-Indy-20160418_01',
    'odoherty_rtt-Indy-20160419_01',
    'odoherty_rtt-Indy-20160420_01',
    'odoherty_rtt-Indy-20160426_01'
]

for dataset_name in datasets:
    context = context_registry.query(alias=dataset_name)
    default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
    default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
    default_cfg.bin_size_ms = 20
    default_cfg.max_arrays = min(max(1, len(context.array)), 2)
    default_cfg.datasets = [context.alias]
    dataset = SpikingDataset(default_cfg)
    dataset.build_context_index()
    trial = 0
    print(dataset[trial]['channel_counts'].sum(1))
# 262...
# print(dataset[trial]['channel_counts'].sum(1))

# 209 is min, 262 is max (and test sesison is 262)
r"""
    Few possible hypotheses, increasing level of difficulty to address

    zero-shot fails because we're unsorted (Let's vet this) and data is more heterogeneous
    zero-shot fails because we're in new "range" of # of channels (this is best case and "easy" to ignore)
    zero-shot fails because we're we have differing number of channels
        - because of a bug
        - because model hones in # of channels as a defining feature for overfitting
        - ?
        [] Look into this by making 2 sets of differing max channels. No reason model should catastrophically overfit in this case, so if it fails, we know it's a bug
"""