#%%
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd
from pathlib import Path

import logging

import pynwb
from pynwb import TimeSeries, ProcessingModule
from pynwb.core import MultiContainerInterface

from nlb_tools.make_tensors import make_train_input_tensors, PARAMS, _prep_mask, make_stacked_array

## Load dataset
path = Path("./data/churchland_reaching/000070/sub-Jenkins/")
path = Path("./data/churchland_reaching/000070/sub-Nitschke/")
# path = Path("./data/nlb/000140/sub-Jenkins/")
exps = list(path.glob("*.nwb"))
exp = exps[0]
# exp = exps[1]
print(exp)

# TODO want to add array group info to units

patch_name = 'churchland_reaching'

class NWBDatasetChurchland(NWBDataset):
    def __init__(self, *args, **kwargs):
        kwargs['split_heldout'] = False
        kwargs['skip_fields'] = [
            'Position_Cursor',
            'Position_Eye',
            'Position_Hand',
            'Processed_A001',
            'Processed_A002',
            'Processed_B001',
            'Processed_B002',
        ]
        # Note - currently these fields are dropped due to a slight timing mismatch.
        # If you want them back, you'll need to reduce precision in NWBDataset.load() from 6 digits to 3 digits (which I think is fine)
        # But we currently don't need
        super().__init__(*args, **kwargs)
        self.trial_info = self.trial_info.rename({ # match NLB naming
            'move_begins_time': 'move_onset_time',
            'task_success': 'success',
            'target_presentation_time': 'target_on_time',
            'reaction_time': 'rt',
        }, axis=1)

dataset = NWBDatasetChurchland(exp) #
bin_width = 5
dataset.resample(bin_width)

# make_tensors from NLB can be used on this data with a few patches
# 1. Params are defined in module, rather than taken as an argument. Override this params
# PARAMS[patch_name] = PARAMS['mc_maze']

# 2. Provide a mock heldout spikes field
# I prefer to override the function as below - unclear how to mock heldout spikes
# 3. Add mock trial_split info
dataset.trial_info['split'] = 'train'

def make_input_tensors_simple(dataset, mock_dataset='mc_maze', trial_split=['train'], **kwargs):
    # See `make_train_input_tensors` for documentation
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"

    # Fetch and update params
    params = PARAMS[mock_dataset].copy()
    # unpack params
    spk_field = params['spk_field']
    # hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()

    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Make output spiking arrays and put into data_dict
    train_dict = make_stacked_array(dataset, [spk_field], make_params, trial_mask)
    return {
        'train_spikes_heldin': train_dict[spk_field]
    }

import pdb;pdb.set_trace()
spikes = make_input_tensors_simple(
    dataset
)
print(spikes.shape)

# train_dict = make_train_input_tensors(
#     dataset,
#     dataset_name='churchland_reaching',
#     trial_split=['train'],
#     save_file=False
# )

# print(train_dict['train_spikes_heldin'].shape)

