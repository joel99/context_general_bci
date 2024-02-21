r"""
- rename associated tasks, loaders to Falcon H1
"""

from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate

import logging

logger = logging.getLogger(__name__)

try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

# Load nwb file
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.01,
        bin_end_timestamps: np.ndarray | None = None
    ) -> np.ndarray:
    r"""
        units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
        bin_end_timestamps: array of timestamps indicating end of bin

        Returns:
        - array of spike counts per bin, per unit. Shape is (bins x units)
    """
    if bin_end_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
    spike_arr = np.zeros((len(bin_end_timestamps), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([-np.inf]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        spike_arr[:, idx] = spike_cnt
    return spike_arr

def load_nwb(fn: str):
    r"""
        Load NWB for Human Motor ARAT dataset. Kinematic timestamps are provided at 100Hz.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        # print(nwbfile)
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematics'].data[:]
        trial_num = nwbfile.acquisition["TrialNum"].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].timestamps[:]
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        epochs = nwbfile.epochs.to_dataframe()
        # Mark pretrial period as first trial - it's a small buffer
        nan_mask = np.isnan(trial_num)
        # Assert nans only occur at start of array, and are only a few
        assert np.all(nan_mask[:np.argmax(~nan_mask)])
        assert np.sum(nan_mask) < 10
        trial_num[np.isnan(trial_num)] = 1

        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            trial_num,
            timestamps,
            epochs,
            labels
        )

def load_files(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    binned, kin, trial_num, timestamps, epochs, labels = zip(*[load_nwb(str(f)) for f in files])
    # Merge data by simple concat
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    trial_num = np.concatenate(trial_num, axis=0)

    # Offset timestamps and epochs to be continuous across multiple datasets
    all_timestamps = [timestamps[0]]
    for idx, current_times in enumerate(timestamps[1:]):
        epochs[idx]['start_time'] += all_timestamps[-1][-1] + 0.01 # 1 bin
        epochs[idx]['stop_time'] += all_timestamps[-1][-1] + 0.01 # 1 bin
        all_timestamps.append(current_times + all_timestamps[-1][-1] + 0.01)
    timestamps = np.concatenate(all_timestamps, axis=0)
    epochs = pd.concat(epochs, axis=0)
    for l in labels[1:]:
        assert l == labels[0]
    return binned, kin, trial_num, timestamps, epochs, labels[0]

@ExperimentalTaskRegistry.register
class FalconLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.falcon

    BASE_RES = 1000 # hz (i.e. 1ms)

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        **kwargs,
    ):
        assert cfg.bin_size_ms == 10, "FALCON data needs 10ms"
        # Load data
        binned, kin, trials, timestamps, epochs, labels = load_files([datapath])
        nan_mask = np.isnan(trials)

        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        for trial_id in np.unique(trials):
            trial_spikes = binned[trials == trial_id]
            trial_vel = kin[trials == trial_id]
            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg, spike_bin_size_ms=10),
                DataKey.bhvr_vel: torch.tensor(trial_vel.copy())
            }
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)