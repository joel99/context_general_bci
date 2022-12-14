from typing import Dict, List, Any
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from einops import rearrange

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors,
    make_eval_input_tensors,
    PARAMS,
    _prep_mask,
    make_stacked_array
)

from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.core import MultiContainerInterface, NWBDataInterface

from config import DataKey, DatasetConfig, MetaKey
from subjects import SubjectInfo, SubjectName, SubjectArrayRegistry
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
TrialNum = int
MetadataKey = str


import logging

logger = logging.getLogger(__name__)

# Core loading strategy pulled from https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb

class NLBLoader(ExperimentalTaskLoader):
    name = "nlb_base"

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        phase='test',
        dataset_cls=NWBDataset,
        make_tensor_fn=make_train_input_tensors,
        make_test_tensor_fn=make_eval_input_tensors,
    ):
        r"""
            Loader for motor tasks in Neural Latents Benchmark (NLB) dataset.
        """
        dataset = dataset_cls(datapath)
        dataset.resample(cfg.bin_size_ms)

        # Create suffix for group naming later
        # suffix = '' if (cfg.bin_size_ms == 5) else f'_{int(cfg.bin_size_ms)}'
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_tensor_fn(
            dataset,
            dataset_name=dataset_alias,
            trial_split=train_split,
            save_file=False
        )

        test_dict = make_test_tensor_fn(
            dataset,
            dataset_name=dataset_alias,
            trial_split='test',
            save_file=False,
        )

        # Show fields of returned dict
        # print(train_dict.keys())

        # Unpack data
        train_spikes_heldin = train_dict['train_spikes_heldin']
        train_spikes_heldout = train_dict['train_spikes_heldout']
        test_spikes_heldin = test_dict['eval_spikes_heldin']
        # Print 3d array shape: trials x time x channel
        # print(train_spikes_heldin.shape)
        train_spikes_heldin = torch.tensor(train_spikes_heldin, dtype=torch.uint8)
        train_spikes_heldout = torch.tensor(train_spikes_heldout, dtype=torch.uint8)
        test_spikes_heldin = torch.tensor(test_spikes_heldin, dtype=torch.uint8)
        meta_payload = {}
        meta_payload['path'] = []
        meta_payload['split'] = []

        arrays_to_use = context_arrays

        for trial in range(train_spikes_heldin.shape[0]):
            spikes = rearrange(train_spikes_heldin[trial], 't c -> t c 1')
            heldout_spikes = rearrange(train_spikes_heldout[trial], 't c -> t c 1')
            spike_payload = {}
            for a in arrays_to_use:
                array = SubjectArrayRegistry.query_by_array(a)
                if array.is_exact:
                    array = SubjectArrayRegistry.query_by_array_geometric(a) # get some typing
                    spike_payload[a] = spikes[:, array.as_indices()].clone()
                else:
                    assert len(arrays_to_use) == 1, "Can't use multiple arrays with non-exact arrays"
                    spike_payload[a] = spikes.clone()
            single_payload = {
                DataKey.spikes: spike_payload,
                DataKey.heldout_spikes: heldout_spikes.clone()
            }
            single_path = cache_root / f"{trial}.pth"
            meta_payload['path'].append(single_path)
            meta_payload['split'].append('train')
            torch.save(single_payload, single_path)
        trial_count = train_spikes_heldin.shape[0]
        for trial in range(test_spikes_heldin.shape[0]):
            spikes = rearrange(test_spikes_heldin[trial], 't c -> t c 1')
            spike_payload = {}
            for a in arrays_to_use:
                array = SubjectArrayRegistry.query_by_array(a)
                if array.is_exact:
                    array = SubjectArrayRegistry.query_by_array_geometric(a)
                    spike_payload[a] = spikes[:, array.as_indices()].clone()
                else:
                    assert len(arrays_to_use) == 1, "Can't use multiple arrays with non-exact arrays"
                    spike_payload[a] = spikes.clone()
            single_payload = {
                DataKey.spikes: spike_payload,
            }
            single_path = cache_root / f"test_{trial_count + trial}.pth"
            meta_payload['path'].append(single_path)
            meta_payload['split'].append('test')
            torch.save(single_payload, single_path)

        return pd.DataFrame(meta_payload)

@ExperimentalTaskRegistry.register
class MazeLoader(NLBLoader):
    name = ExperimentalTask.nlb_maze

@ExperimentalTaskRegistry.register
class RTTLoader(NLBLoader):
    name = ExperimentalTask.nlb_rtt


# === Overrides for Churchland dataset ===
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

        self.trial_info['split'] = 'train'

    def load(self, fpath, split_heldout=True, skip_fields=[]):
        """Loads data from an NWB file into two dataframes,
        one for trial info and one for time-varying data.

        # ! Overriden to process Churchland dataset more carefully
        # ! Specifically Churchland release units don't appear to all be restrained to `obs_intervals` or even trial times
        # ! Skips loading of other timeseries

        Parameters
        ----------
        fpath : str
            Path to the NWB file
        split_heldout : bool, optional
            Whether to load heldin units and heldout units
            to separate fields or not, by default True
        skip_fields : list, optional
            List of field names to skip during loading,
            which may be useful if memory is an issue.
            Field names must match the names automatically
            assigned in the loading process. Spiking data
            can not be skipped. Field names in the list
            that are not found in the dataset are
            ignored

        Returns
        -------
        tuple
            Tuple containing a pd.DataFrame of continuous loaded
            data, a pd.DataFrame with trial metadata, a dict
            with descriptions of fields in the DataFrames, and
            the bin width of the loaded data in ms
        """
        logger.info(f"Loading {fpath}")
        import pdb;pdb.set_trace()
        # Open NWB file
        io = NWBHDF5IO(fpath, 'r')
        nwbfile = io.read()

        # Load trial info and units
        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({'id': 'trial_id', 'stop_time': 'end_time'}, axis=1))
        units = nwbfile.units.to_dataframe()
        # Load descriptions of trial info fields
        descriptions = {}
        for name, info in zip(nwbfile.trials.colnames, nwbfile.trials.columns):
            descriptions[name] = info.description

        # Find all timeseries
        def make_df(ts):
            """Converts TimeSeries into pandas DataFrame"""
            if ts.timestamps is not None:
                index = ts.timestamps[()]
            else:
                index = np.arange(ts.data.shape[0]) / ts.rate + ts.starting_time
            columns = ts.comments.split('[')[-1].split(']')[0].split(',') if 'columns=' in ts.comments else None
            df = pd.DataFrame(ts.data[()], index=pd.to_timedelta(index, unit='s'), columns=columns)
            return df

        def find_timeseries(nwbobj):
            """Recursively searches the NWB file for time series data"""
            ts_dict = {}
            for child in nwbobj.children:
                if isinstance(child, TimeSeries):
                    if child.name in skip_fields:
                        continue
                    ts_dict[child.name] = make_df(child)
                    descriptions[child.name] = child.description
                elif isinstance(child, ProcessingModule):
                    pm_dict = find_timeseries(child)
                    ts_dict.update(pm_dict)
                elif isinstance(child, MultiContainerInterface):
                    for field in child.children:
                        if isinstance(field, TimeSeries):
                            name = child.name + "_" + field.name
                            if name in skip_fields:
                                continue
                            ts_dict[name] = make_df(field)
                            descriptions[name] = field.description
            return ts_dict
        # Create a dictionary containing DataFrames for all time series
        # data_dict = find_timeseries(nwbfile)
        data_dict = {}
        # Calculate data index
        start_time = 0.0
        bin_width = 1 # in ms, this will be the case for all provided datasets
        rate = round(1000. / bin_width, 2) # in Hz
        # Use obs_intervals, or last trial to determine data end

        # In Churchland release for Monkey J datasets, trial times are occassionally reset.
        # We hope it happens fairly infrequently in the following in-place patch
        trial_info['discard_trial_patch'] = False
        def scan_and_adjust_times(trial_info, units, discontinuity_pad_s=1.0):
            # fields to update in units - `obs_intervals` and `spike_times`
            spike_idx = np.zeros(len(units), dtype=int) # spike_times aren't grouped by trial, so we need to estimate which trial we're in
            for i in range(1, len(trial_info)):
                # offset = 0
                if trial_info['start_time'].iloc[i] < trial_info['end_time'].iloc[i-1]:
                    print(f"Discontinuity detected at trial {i} of {len(trial_info)}, discard rest!")
                    trial_info.loc[i:, 'discard_trial_patch'] = True
                    for j in range(len(units)):
                        units.spike_times[j] = units.spike_times[j][:spike_idx[j]]
                        units.obs_intervals[j] = units.obs_intervals[j][:i]
                    trial_info = trial_info[:i]
                    break
                # It's not actually clear what the nature of the discontinuity is, so I can't patch
                #     offset = trial_info['end_time'].iloc[i-1] - trial_info['start_time'].iloc[i] + discontinuity_pad_s
                #     # Current behavior is to shift all subsequent trials by the difference between the end of the previous trial and the start of the current trial
                #     # However, for small offsets; this might just be actually overlapping trials; we try to skip this
                #     # Hopefully downstream processes take care of those offsets
                #     if offset < 10.0:
                #         offset = 0
                #     else:
                #         for k in trial_info.columns:
                #             if k.endswith('_time'):
                #                 trial_info.loc[i:, k] = trial_info.loc[i:, k] + offset
                #                 # trial_info[k][i:] = trial_info[k][i:] + offset
                for j in range(len(units)):
                    unit_times = units.spike_times[j]
                    # if offset:
                    #     import pdb;pdb.set_trace()
                    #     units.obs_intervals[j][i:] += offset # shape Trials x 2
                    #     if spike_idx[j] + 1 < len(unit_times): # spike_idx[j] tracks the last spike in the previous trial
                    #         unit_times[spike_idx[j] + 1:] += offset
                    # Update pointer to one past last spike in current trial
                    while spike_idx[j] + 1 < len(unit_times) and \
                        (unit_times[spike_idx[j]] < trial_info['end_time'].iloc[i]) and \
                        (unit_times[spike_idx[j]+1] + 10.0 > unit_times[spike_idx[j]]): # kind of blunt but there's going to be a discontunuity next iteration
                        # (unit_times[spike_idx[j]] >= trial_info['start_time'].iloc[i-1]): # one option - not good if we have spikes outside trials (even if those are discarded)
                        # Fine that we don't iterate to end - we just want to update times here
                        # + 10 for same offset
                        spike_idx[j] += 1
        # ! Note we don't (for unsupervised pretraining) consider whether to discard trials according to `discard_trial` flag
        # ! But we should track it if we are interested in decoding kinematics

        scan_and_adjust_times(trial_info, units)
        # import pdb;pdb.set_trace()

        end_time = round(max(units.obs_intervals.apply(lambda x: x[-1][-1])) * rate) * bin_width
        if (end_time < trial_info['end_time'].iloc[-1]):
            print("obs_interval ends before trial end") # TO REMOVE
            end_time = round(trial_info['end_time'].iloc[-1] * rate) * bin_width
        timestamps = (np.arange(start_time, end_time, bin_width) / 1000).round(6)
        timestamps_td = pd.to_timedelta(timestamps, unit='s')

        # Check that all timeseries match with calculated timestamps
        for key, val in list(data_dict.items()):
            if not np.all(np.isin(np.round(val.index.total_seconds(), 6), timestamps)):
                logger.warning(f"Dropping {key} due to timestamp mismatch.")
                data_dict.pop(key)

        def make_mask(obs_intervals):
            """Creates boolean mask to indicate when spiking data is not in obs_intervals"""
            mask = np.full(timestamps.shape, True)
            for start, end in obs_intervals:
                start_idx = np.ceil(round((start - timestamps[0]) * rate, 6)).astype(int)
                end_idx = np.floor(round((end - timestamps[0]) * rate, 6)).astype(int)
                mask[start_idx:end_idx] = False
            return mask

        # Prepare variables for spike binning
        masks = [(~units.heldout).to_numpy(), units.heldout.to_numpy()] if split_heldout else [np.full(len(units), True)]

        for mask, name in zip(masks, ['spikes', 'heldout_spikes']):
            # Check if there are any units
            if not np.any(mask):
                continue

            # Allocate array to fill with spikes
            spike_arr = np.full((len(timestamps), np.sum(mask)), 0.0, dtype='float16')

            # Bin spikes using decimal truncation and np.unique - faster than np.histogram with same results
            for idx, (_, unit) in enumerate(units[mask].iterrows()):
                # spike_times = unit.spike_times
                # if np.argmin(spike_times) != 0: # few weird units
                #     logger.warning(f'Reordering spike times in unit {idx}')
                #     unit.spike_times = np.sort(spike_times)
                spike_idx, spike_cnt = np.unique(((unit.spike_times - timestamps[0]) * rate).round(6).astype(int), return_counts=True)
                # JY patch: restrict to timestamps that are feasible for allocated spike_arr (which is defined wrt obs_interval, which is hopefully defined wrt trials)
                in_experiment_idx_mask = np.logical_and(spike_idx >= 0, spike_idx < spike_arr.shape[0])
                if not np.all(in_experiment_idx_mask):
                    import pdb;pdb.set_trace() # this shouldn't ever happen...
                    logger.warning(f'Restricting spike times in unit {unit.id}')
                spike_idx = spike_idx[in_experiment_idx_mask]
                spike_cnt = spike_cnt[in_experiment_idx_mask]
                spike_arr[spike_idx, idx] = spike_cnt

            # Replace invalid intervals in spike recordings with NaNs
            if 'obs_intervals' in units.columns:
                neur_mask = make_mask(units[mask].iloc[0].obs_intervals)
                if np.any(spike_arr[neur_mask]):
                    logger.warning(f"{spike_arr[neur_mask].sum()} Spikes found outside of observed interval.")
                spike_arr[neur_mask] = np.nan

            # Create DataFrames with spike arrays
            data_dict[name] = pd.DataFrame(spike_arr, index=timestamps_td, columns=units[mask].index).astype('float16', copy=False)

        # Create MultiIndex column names
        data_list = []
        for key, val in data_dict.items():
            chan_names = None if type(val.columns) == pd.RangeIndex else val.columns
            val.columns = self._make_midx(key, chan_names=chan_names, num_channels=val.shape[1])
            data_list.append(val)

        # Assign time-varying data to `self.data`
        data = pd.concat(data_list, axis=1)
        data.index.name = 'clock_time'
        data.sort_index(axis=1, inplace=True)

        # Convert time fields in trial info to timedelta
        # and assign to `self.trial_info`
        def to_td(x):
            if x.name.endswith('_time'):
                return pd.to_timedelta(x, unit='s')
            else:
                return x
        trial_info = trial_info.apply(to_td, axis=0)

        io.close()

        return data, trial_info, descriptions, bin_width


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

@ExperimentalTaskRegistry.register
class ChurchlandMazeLoader(NLBLoader):
    name = ExperimentalTask.churchland_maze

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        phase='test',
        dataset_cls=NWBDatasetChurchland,
        make_tensor_fn=make_input_tensors_simple
    ):
        return super().load(
            datapath,
            cfg,
            cache_root,
            subject,
            context_arrays,
            dataset_alias,
            phase,
            dataset_cls,
            make_tensor_fn
        )