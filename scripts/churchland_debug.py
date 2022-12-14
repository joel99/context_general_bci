#%%
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd
from pathlib import Path

import logging

import pynwb
from pynwb import TimeSeries, ProcessingModule, NWBFile, NWBHDF5IO
from pynwb.core import MultiContainerInterface

from nlb_tools.make_tensors import make_train_input_tensors, PARAMS, _prep_mask, make_stacked_array
from contexts import context_registry

## Load dataset
# From this investigation, it seems like unit 1 is busted
dataset_name = 'churchland_maze_jenkins-0'
# dataset_name = 'churchland_maze_jenkins-3'
dataset_name = 'churchland_maze_nitschke-3'
context = context_registry.query(alias=dataset_name)
path = context.datapath
print(context.datapath)
io = NWBHDF5IO(context.datapath, 'r')
nwbfile = io.read()

#%%
from matplotlib import pyplot as plt
# print(nwbfile.fields.keys())
# print(nwbfile.subject)

# TODO explore datafile
# Has movement and LFP
# print(nwbfile.processing['behavior'].data_interfaces)
# print(nwbfile.processing['ecephys'].data_interfaces)

# Time intervals: https://pynwb.readthedocs.io/en/stable/tutorials/general/file.html#time-intervals
# print(nwbfile.intervals['trials'])
# Actually, this is the same thing as nwbfile.trials
print(nwbfile.intervals['trials'].colnames)
print(nwbfile.intervals['trials']['start_time'][:])
print(nwbfile.intervals['trials']['go_cue_time'][:])
print(nwbfile.intervals['trials']['move_begins_time'][:])
print(nwbfile.intervals['trials']['move_ends_time'][:])
print(nwbfile.intervals['trials']['discard_trial'][:].sum())
# print(nwbfile.intervals['trials']['stop_time'][:])
# print(len(nwbfile.intervals['trials']['start_time'][:]))
# print(len(nwbfile.intervals['trials']['stop_time'][:]))
# plt.plot(nwbfile.intervals['trials']['start_time'][:])

# More time intervals?
# print(nwbfile.trials.id)

#%%
unit = 2
unit = 20
# unit = 1
# print(nwbfile.units)
# print(nwbfile.units.spike_times)
# print(nwbfile.units.to_dataframe().obs_intervals.iloc[unit])
# plt.plot(nwbfile.units.to_dataframe().obs_intervals.iloc[unit][:,0])
# plt.plot(nwbfile.units.to_dataframe().obs_intervals.iloc[unit][:,1])
# plt.plot(nwbfile.units.to_dataframe().spike_times.iloc[unit])
# print(nwbfile.units.to_dataframe().spike_times.iloc[1])
# spikes = nwbfile.units.to_dataframe().spike_times.iloc[unit]
is_monotonic = lambda spikes: np.all(np.diff(spikes) >= 0)
all_units = nwbfile.units.to_dataframe()
unit_valid = []
for t in range(len(all_units)):
    unit_valid.append(is_monotonic(all_units.spike_times.iloc[t]))
print((np.array(unit_valid) == False).nonzero())

#%%
# Sanity check interval and spike time validity
intervals = nwbfile.units.to_dataframe().obs_intervals.iloc[unit]
print(len(intervals))
#%%
num_contained = []
for t in nwbfile.units.to_dataframe().spike_times.iloc[unit]:
    num_contained.append(np.sum((intervals[:,0] <= t) & (intervals[:,1] >= t)))

print((np.array(num_contained) == 1).all())

# Identify number of overlapping intervals in `intervals`
def num_overlaps(intervals):
    num_overlaps = 0
    for i in range(len(intervals)):
        for j in range(i+1, len(intervals)):
            if (intervals[i,0] <= intervals[j,0] <= intervals[i,1]) or (intervals[j,0] <= intervals[i,0] <= intervals[j,1]):
                num_overlaps += 1
    return num_overlaps
print(num_overlaps(intervals))







#%%
# io.close()
# exit(0)
#%%
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

