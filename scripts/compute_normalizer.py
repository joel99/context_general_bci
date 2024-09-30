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
# task_query = 'M2'
# task_query = 'calib_odoherty_calib_rtt'
# task_query = 'calib_pitt_calib_broad'
task_query = 'calib_pitt_grasp'
task_query = 'calib_cst_calib'
# task_query = 't5'

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.bin_size_ms = 20
default_cfg.odoherty_rtt.chop_size_ms = 2000
default_cfg.odoherty_rtt.include_sorted = False
default_cfg.odoherty_rtt.arrays = ["Indy-M1", "Loco-M1"]
# default_cfg.max_length_ms = 4000
default_cfg.datasets = [f'{task_query}.*']
if task_query == 't5':
    default_cfg.datasets = [
        't5_06_02_2021',
        't5_06_04_2021',
        't5_06_23_2021',
        't5_06_28_2021',
        't5_06_30_2021',
        't5_07_12_2021',
        't5_07_14_2021',
        't5_10_11_2021',
        't5_10_13_2021'
    ]
default_cfg.pitt_co.respect_trial_boundaries = True


dataset = SpikingDataset(default_cfg)
print(len(dataset))
dataset.build_context_index()

all_kin = []
dims = 0
for i in range(len(dataset)):
    # check for nans in spikes and bhvr_vel while we're here.
    if torch.isnan(dataset[i][DataKey.spikes]).any():
        print(f'NaN spikes in trial {i}')
        breakpoint()
    if torch.isnan(dataset[i][DataKey.bhvr_vel]).any():
        print(f'NaN bhvr_vel in trial {i}')
        breakpoint()
    trial_vel = dataset[i][DataKey.bhvr_vel]
    dims = trial_vel.shape[1]
    all_kin.append(trial_vel)
print(f'Dims: {dims}')
all_kin = torch.cat(all_kin, 0)
kin_mean = all_kin.mean(0).float()
kin_std = all_kin.std(0).float()

print(f'Mean: {kin_mean}')
print(f'Std: {kin_std}')
torch.save({
    'mean': kin_mean,
    'std': kin_std,
}, f'./data/zscore_{task_query.lower()}_{kin_mean.shape[0]}d.pt')
# %%
