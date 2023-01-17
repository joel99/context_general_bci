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

# Note these comrpise a bunch of different tasks, perhaps worth denoting/splitting them
gdown_ids = [
    # Jenkins, milestone 1 9/2015 -> 1/2016 (vs all in 2009 for DANDI)
    'https://drive.google.com/file/d/1o3X-L7uFH0vVPollVaD64AmDPwc0kXkq/view?usp=share_link',
    'https://drive.google.com/file/d/1MmnXvAMSBvt_eZ8X-CgmOWibHqnk_1vr/view?usp=share_link',
    'https://drive.google.com/file/d/10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI/view?usp=share_link',
    'https://drive.google.com/file/d/1msGk3H6yPwS4GCcJwZybJFFX6JWcbvYp/view?usp=share_link',
    'https://drive.google.com/file/d/16QJZXe0xQKIVqWV5XhOPDIzPBX1V2CV4/view?usp=share_link',
    'https://drive.google.com/file/d/1pe3gnurM4xY5R9qGQ8ohMi1h2Lv-eJWf/view?usp=share_link',
    'https://drive.google.com/file/d/16QJZXe0xQKIVqWV5XhOPDIzPBX1V2CV4/view?usp=share_link',
    'https://drive.google.com/file/d/1Uxht3GUFdJ9Y0AcyTYfCp7uhwCvX0Ujs/view?usp=share_link',
    'https://drive.google.com/file/d/1hxD7xKu96YEMD8iTuHVF6mSAv-h5xlxG/view?usp=share_link',

    # Reggie, milestone 1 ~ 2017
    '151nE5p4OTSwiR7UyW2s9RBMGklYLvYO1',
    '1TFVbWjdTgQ4XgfiRN3ilwfya4LAk9jgB',
    '1m8YxKehWhZlkFn9p9XKk8bWnIhfLy1ja',
    '1-qq1JiOOChq80xasEhtkwUCP3j_b2_v1',
    '1413W9XGLJ2gma1CCXpg1DRDGpl4-uxkG',
    '19euCNYTHipP7IJTGBPtu-4efuShW6qSk',
    '1eePWeHohrhbtBwQg8fJJWjPJDCtWLV3S',

    # Nitschke (9/22-28/2010)
    'https://drive.google.com/file/d/1IHPADrDpwdWEZKVjC1B39NIf_FdSv49k/view?usp=share_link',
    'https://drive.google.com/file/d/1D8KYfy5IwMmEZaKOEv-7U6-4s-7cKINK/view?usp=share_link',
    'https://drive.google.com/file/d/1tp_ezJqvgW5w_e8uNBvrdbGkgf2CP_Sj/view?usp=share_link',
    'https://drive.google.com/file/d/1Im75cmAPuS2dzHJUGw9v5fuy9lx49r6c/view?usp=share_link',
    # skip 9/22 provided in DANDI release
]

if __name__ == '__main__':
    # Pull the various files using `gdown` (pip install gdown)
    # https://github.com/catalystneuro/shenoy-lab-to-nwb
    # -- https://drive.google.com/drive/folders/1mP3MCT_hk2v6sFdHnmP_6M0KEFg1r2g_
    import gdown

    for gid in gdown_ids:
        if gid.startswith('http'):
            gid = gid.split('/')[-2]
        if not Path(f'./data/churchland_misc/{gid}.mat').exists():
            gdown.download(id=gid, output=f'./data/churchland_misc/{gid}.mat', quiet=False)

@ExperimentalTaskRegistry.register
class ChurchlandMiscLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.churchland_misc
    r"""
    Churchland/Kaufman reaching data, from gdrive. Assorted extra sessions that don't overlap with DANDI release.

    List of IDs
    # - register, make task etc

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
            is_valid = ~(trial_info['discard_trial'][:].astype(bool))
            move_begins = trial_info['move_begins_time'][:]
            move_ends = trial_info['move_ends_time'][:]
            trial_ends = trial_info['stop_time'][:]
            end_time_mapped = np.isnan(move_ends)
            move_ends[end_time_mapped] = trial_ends[end_time_mapped]
            spikes = nwbfile.units.to_dataframe()
            spike_intervals = spikes.obs_intervals # 1 per unit
            spike_times = spikes.spike_times # 1 per unit

            move_begins = move_begins[is_valid]
            move_ends = move_ends[is_valid]
            for t in range(len(spike_intervals)):
                spike_intervals.iloc[t] = spike_intervals.iloc[t][is_valid]

        meta_payload = {}
        meta_payload['path'] = []

        drop_units = BLACKLIST_UNITS
        # Validation
        def is_ascending(times):
            return np.all(np.diff(times) > 0)
        reset_time = 0
        if not is_ascending(move_begins):
            first_nonascend = np.where(np.diff(move_begins) <= 0)[0][0]
            logger.warning(f"Move begins not ascending, cropping to ascending {(100 * first_nonascend / len(move_begins)):.2f} %")
            move_begins = move_begins[:first_nonascend]
            move_ends = move_ends[:first_nonascend]
            is_valid = is_valid[:first_nonascend]
            reset_time = move_begins[-1]
            # No need to crop obs_intervals, we'll naturally only index so far in
        for mua_unit in range(len(spike_times)):
            if not is_ascending(spike_times.iloc[mua_unit]) and mua_unit not in drop_units:
                reset_idx = (spike_times.iloc[mua_unit] > reset_time).nonzero()
                if len(reset_idx) > 0:
                    reset_idx = reset_idx[0][0]
                    logger.warning(f"Spike times for unit {mua_unit} not ascending, crop to {(100 * reset_idx / len(spike_times.iloc[mua_unit])):.2f}% of spikes")
                    spike_times.iloc[mua_unit] = spike_times.iloc[mua_unit][:reset_idx]
                else:
                    logger.warning(f"No reset index found for unit {mua_unit}! Skipping...")
                    drop_units.append(mua_unit)
                # Based on explorations, several of the datasets have repeated trials / unit times. All appear to complete/get to a fairly high point before resetting


        for t in range(len(spike_intervals)):
            assert (spike_intervals.iloc[t] == spike_intervals.iloc[0]).all(), "Spike intervals not equal"
        spike_intervals = spike_intervals.iloc[0] # all equal in MUA recordings
        # Times are in units of seconds

        arrays_to_use = context_arrays
        assert len(spike_times) == 192, "Expected 192 units"
        for t in range(len(move_begins)):
            # if not is_valid[t]:
            #     continue # we subset now
            if t > 0 and spike_intervals[t][0] < spike_intervals[t-1][1]:
                logger.warning(f"Observation interval for trial {t} overlaps with previous trial, skipping...")
                continue

            start, end = move_begins[t] - cfg.churchland_maze.pretrial_time_s, move_ends[t] + cfg.churchland_maze.posttrial_time_s
            if start <= spike_intervals[t][0]:
                logger.warning("Movement begins before observation interval, cropping...")
                start = spike_intervals[t][0]
            if end > spike_intervals[t][1]:
                if not end_time_mapped[t]: # will definitely be true if end time mapped
                    logger.warning(f"Movement ends after observation interval, cropping... (diff = {(end - spike_intervals[t][1]):.02f}s)")
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
