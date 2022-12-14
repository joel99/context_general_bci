#%%
from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import h5py
from scipy.interpolate import interp1d

from config import DataKey, DatasetConfig
from subjects import SubjectInfo, SubjectArrayRegistry
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from einops import rearrange, reduce

import logging

logger = logging.getLogger(__name__)

@ExperimentalTaskRegistry.register
class ChurchlandMazeLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.churchland_maze
    r"""
    Churchland/Kaufman reaching data.
    # https://dandiarchive.org/dandiset/000070/draft/files

    We write a slightly different loader rather than use NLB loader
    for a bit more granular control.
    """

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        sampling_rate: int = 1000 # Hz
    ):
        assert cfg.churchland_maze.chop_size_ms % cfg.bin_size_ms == 0, "Chop size must be a multiple of bin size"
        assert cfg.odoherty_rtt.load_covariates == False, "Covariates not supported yet"
        with h5py.File(datapath, 'r') as h5file:
            orig_timestamps = np.squeeze(h5file['t'][:])
            time_span = int((orig_timestamps[-1] - orig_timestamps[0]) * sampling_rate)
            if cfg.odoherty_rtt.load_covariates:
                def upsample(array, factor, kind='cubic'):
                    if factor > 1:
                        ip_fn = interp1d(np.arange(array.shape[0]), array, kind=kind, fill_value='extrapolate', axis=0)
                        upsampled = ip_fn(np.arange(0, array.shape[0]-1, round(1 / factor, 4)))
                    else:
                        upsampled = array
                    return upsampled

                # sampled at 250Hz, upsample to 1000Hz to match 1ms bin width
                # TODO proper down interpolation? Nah, low priority
                target_rate = round(0.001 / cfg.bin_size_ms)
                upsample_factor = target_rate / 250
                finger_pos = upsample(h5file['finger_pos'][()].T, 4)
                cursor_pos = upsample(h5file['cursor_pos'][()].T, 4)
                target_pos = upsample(h5file['target_pos'][()].T, 4, kind='previous')
                cursor_vel = np.gradient(cursor_pos[~np.isnan(cursor_pos[:, 0])], axis=0)
                raise NotImplementedError #  Need to do something useful (just stack it with spikes)

            int_arrays = [h5file[ref][()][:,0] for ref in h5file['chan_names'][0]]
            make_chan_name = lambda array: ''.join([chr(num) for num in array])
            chan_names = [make_chan_name(array) for array in int_arrays]
            chan_arrays = [cn.split()[0] for cn in chan_names]
            assert (
                len(chan_arrays) == 96 and all([c == 'M1' for c in chan_arrays]) or \
                len(chan_arrays) == 192 and all([c == 'M1' for c in chan_arrays[:96]]) and all([c == 'S1' for c in chan_arrays[96:]])
            ), "Only M1 and S1 arrays in specific format are supported"

            spike_refs = h5file['spikes'][()].T
            channels, units = spike_refs.shape # units >= 1 are sorted, we just want MUA on unit 0
            mua_unit = 0

            spike_arr = torch.zeros((time_span, channels), dtype=torch.uint8)
            min_spike_time = []
            for c in range(channels):
                if h5file[spike_refs[c, mua_unit]].dtype != np.float:
                    continue
                spike_times = np.squeeze(h5file[spike_refs[c, mua_unit]][()], axis=0)
                spike_times = spike_times - orig_timestamps[0]
                ms_spike_times, ms_spike_cnt = np.unique((spike_times * sampling_rate).round(6).astype(int), return_counts=True)
                spike_mask = ms_spike_times < spike_arr.shape[0]
                ms_spike_times = ms_spike_times[spike_mask]
                ms_spike_cnt = torch.tensor(ms_spike_cnt[spike_mask], dtype=torch.uint8)
                spike_arr[ms_spike_times, c] = ms_spike_cnt
                min_spike_time.append(ms_spike_times[0])
        min_spike_time = max(min(min_spike_time), 0) # some spikes come before marked trial start
        spike_arr: torch.Tensor = spike_arr[min_spike_time:, :]
        full_spikes = spike_arr.unfold(
            0, cfg.odoherty_rtt.chop_size_ms, cfg.odoherty_rtt.chop_size_ms
        ) # Trial x C x chop_size (time)
        full_spikes = reduce(
            rearrange(full_spikes, 'b c (time bin) -> b time c bin', bin=cfg.bin_size_ms),
            'b time c bin -> b time c 1', 'sum'
        )

        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        if len(chan_names) == 96 and len(arrays_to_use) > 1:
            logger.error(f'{datapath} only has 96 ch but registered {context_arrays}, update RTTContextInfo')
            return None
            raise NotImplementedError
        for t in range(full_spikes.size(0)):
            spikes = full_spikes[t]
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
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
