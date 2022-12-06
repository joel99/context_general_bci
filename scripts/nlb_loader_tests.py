#%%
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import logging

from contexts import context_registry

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

