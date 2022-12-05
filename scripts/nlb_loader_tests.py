#%%
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py

import logging

from context_registry import context_registry

# dataset_name = 'mc_maze_large'
# datapath = './000138/sub-Jenkins/'
context_registry.query(alias='mc_maze_large')
# dataset = NWBDataset(datapath)
