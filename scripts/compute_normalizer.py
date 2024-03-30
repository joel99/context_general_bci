#%%
# Simple script to compute normalized kin dimensions for a given dataset
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.tasks.pitt_co import CURSOR_TRANSLATE_AND_CLICK
# task_query = 'H1'
# task_query = 'M1'
task_query = 'calib_odoherty_calib_rtt'
task_query = 'calib_pitt_calib_broad'

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.bin_size_ms = 20
default_cfg.odoherty_rtt.chop_size_ms = 2000
default_cfg.odoherty_rtt.include_sorted = False
default_cfg.odoherty_rtt.arrays = ["Indy-M1", "Loco-M1"]
# default_cfg.max_length_ms = 4000
default_cfg.datasets = [f'{task_query}.*']
default_cfg.pitt_co.respect_trial_boundaries = True

dataset = SpikingDataset(default_cfg)
print(len(dataset))
dataset.build_context_index()

all_kin = []
for i in range(len(dataset)):
    trial_vel = dataset[i][DataKey.bhvr_vel]
    all_kin.append(trial_vel)
all_kin = torch.cat(all_kin, 0)
kin_mean = all_kin.mean(0).float()
kin_std = all_kin.std(0).float()

print(kin_mean)
print(kin_std)
torch.save({
    'mean': kin_mean,
    'std': kin_std,
}, f'./data/zscore_{task_query.lower()}_{kin_mean.shape[0]}d.pt')
# %%
