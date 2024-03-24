#%%
# Simple script to compute normalized kin dimensions for a given dataset
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.utils import wandb_query_latest
from context_general_bci.analyze_utils import prep_plt, load_wandb_run

sample_query = 'h1'
wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
default_cfg = cfg.dataset
# default_cfg: DatasetConfig = OmegaConf.create(DatasetConfig())
# default_cfg.data_keys = [DataKey.spikes]
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
dataset = SpikingDataset(default_cfg)
dataset.build_context_index()

all_kin = []
for i in range(len(dataset)):
    trial_vel = dataset[i][DataKey.bhvr_vel]
    all_kin.append(trial_vel)
all_kin = torch.cat(all_kin, 0)
kin_mean = all_kin.mean(0)
kin_std = all_kin.std(0)

print(kin_mean)
print(kin_std)
torch.save({
    'mean': kin_mean,
    'std': kin_std,
}, f'./data/zscore_{sample_query}_{kin_mean.shape[0]}d.pt')
# %%
