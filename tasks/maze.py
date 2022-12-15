#%%
from typing import List
from pathlib import Path
import math
import numpy as np
import torch
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d

from config import DataKey, DatasetConfig
from subjects import SubjectInfo, SubjectArrayRegistry
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from einops import rearrange, reduce

import logging

logger = logging.getLogger(__name__)

BLACKLIST_UNITS = [1]
@ExperimentalTaskRegistry.register
class ChurchlandMazeLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.churchland_maze
    r"""
    Churchland/Kaufman reaching data.
    # https://dandiarchive.org/dandiset/000070/draft/files

    Initial exploration done in `churchland_debug.py`.
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
        if cfg.churchland_maze.chop_size_ms > 0:
            assert cfg.churchland_maze.chop_size_ms % cfg.bin_size_ms == 0, "Chop size must be a multiple of bin size"
            # if 0, no chop, just send in full lengths
        assert cfg.churchland_maze.load_covariates == False, "Covariates not supported yet"

        with NWBHDF5IO(datapath, 'r') as io:
            nwbfile = io.read()
            trial_info = nwbfile.trials
            move_begins = trial_info['move_begins_time'][:]
            move_ends = trial_info['move_ends_time'][:]
            is_valid = ~trial_info['discard_trial'][:]
            spikes = nwbfile.units.to_dataframe()
            spike_intervals = spikes.obs_intervals # 1 per unit
            spike_times = spikes.spike_times # 1 per unit

        meta_payload = {}
        meta_payload['path'] = []

        drop_units = BLACKLIST_UNITS
        # Validation
        def is_ascending(times):
            return np.all(np.diff(times) > 0)
        for mua_unit in range(len(spike_times)):
            if not is_ascending(spike_times.iloc[mua_unit]) and mua_unit not in drop_units:
                drop_units.append(mua_unit)
                logger.warning(f"Spike times for unit {mua_unit} not ascending, blacklisting...")


        for t in range(len(spike_intervals)):
            assert (spike_intervals.iloc[t] == spike_intervals.iloc[0]).all(), "Spike intervals not equal"
        spike_intervals = spike_intervals.iloc[0] # all equal in MUA recordings
        # Times are in units of seconds

        arrays_to_use = context_arrays
        assert len(spike_times) == 192, "Expected 192 units"
        for t in range(len(move_begins)):
            if not is_valid[t]:
                continue
            if t > 0 and spike_intervals[t][0] < spike_intervals[t-1][1]:
                logger.warning(f"Observation interval for trial {t} overlaps with previous trial, skipping...")
                continue

            start, end = move_begins[t] - cfg.churchland_maze.pretrial_time_s, move_ends[t] + cfg.churchland_maze.posttrial_time_s
            if start <= spike_intervals[t][0]:
                logger.warning("Movement begins before observation interval, cropping...")
                start = spike_intervals[t][0]
            if end >= spike_intervals[t][1]:
                logger.warning("Movement ends after observation interval, cropping...")
                end = spike_intervals[t][1]
            if math.isnan(end - start):
                logger.warning(f"Trial {t} has NaN duration, skipping...") # this occurs irreproducibly...
                continue
            time_span = int((end - start) * sampling_rate) + 1 # +1 for rounding error
            trial_spikes = torch.zeros((time_span, len(spike_times)), dtype=torch.uint8)
            for c in range(len(spike_times)):
                if c in drop_units:
                    continue
                unit_times = spike_times.iloc[c]
                unit_times = unit_times[(unit_times >= start) & (unit_times < end)]
                unit_times = unit_times - start
                ms_spike_times, ms_spike_cnt = np.unique(np.floor(unit_times * sampling_rate), return_counts=True)
                trial_spikes[ms_spike_times, c] = torch.tensor(ms_spike_cnt, dtype=torch.uint8)

            # trim to valid length and then reshape
            if cfg.churchland_maze.chop_size_ms > 0:
                trial_spikes = trial_spikes[:cfg.churchland_maze.chop_size_ms]
            elif trial_spikes.size(0) % cfg.bin_size_ms:
                trial_spikes = trial_spikes[:-(trial_spikes.size(0) % cfg.bin_size_ms)]
            trial_spikes = trial_spikes[:cfg.churchland_maze.max_length_ms]
            trial_spikes = rearrange(
                trial_spikes,
                '(t bin) c -> t bin c', bin=cfg.bin_size_ms
            )
            trial_spikes = reduce(trial_spikes, 't bin c -> t c 1', 'sum')
            trial_spikes = trial_spikes.to(dtype=torch.uint8)
            spike_payload = {}
            for a in arrays_to_use:
                array = SubjectArrayRegistry.query_by_array(a)
                if array.is_exact:
                    array = SubjectArrayRegistry.query_by_array_geometric(a)
                    spike_payload[a] = trial_spikes[:, array.as_indices()].clone()
                else:
                    assert len(arrays_to_use) == 1, "Can't use multiple arrays with non-exact arrays"
                    spike_payload[a] = trial_spikes.clone()
            single_payload = {
                DataKey.spikes: spike_payload,
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)