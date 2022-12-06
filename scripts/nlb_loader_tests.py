#%%
# Largely pulled from https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import logging

from context_registry import context_registry

dataset_name = 'mc_maze_large' # 122 sorted units
dataset_name = 'mc_maze_medium' # 114 sorted units
dataset_name = 'mc_maze_small' # 107 sorted units
dataset_name = 'mc_maze' # 137 sorted units
context = context_registry.query(alias=dataset_name)
dataset = NWBDataset(context.datapath)

#%%
# We use our own train/val splits for convenience.
phase = 'test'

# Choose bin width and resample
bin_width = 5
dataset.resample(bin_width)

# Create suffix for group naming later
suffix = '' if (bin_width == 5) else f'_{int(bin_width)}'

train_split = 'train' if (phase == 'val') else ['train', 'val']
train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split=train_split, save_file=False)

# Show fields of returned dict
print(train_dict.keys())

# Unpack data
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']

# Print 3d array shape: trials x time x channel
print(train_spikes_heldin.shape)
