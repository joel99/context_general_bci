r"""
- rename associated tasks, loaders to Falcon H1
"""

from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import logging

logger = logging.getLogger(__name__)

try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig, ExperimentalConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

# Load nwb file
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.01,
        bin_end_timestamps: Optional[np.ndarray] = None
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
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:].astype(dtype=np.float32)
        kin_mask = ~nwbfile.acquisition['Blacklist'].data[:].astype(bool)
        trial_num = nwbfile.acquisition["TrialNum"].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].timestamps[:]
        labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        epochs = nwbfile.epochs.to_dataframe()
        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            kin_mask,
            trial_num,
            timestamps,
            epochs,
            labels
        )

def load_files(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    binned, kin, kin_mask, trial_num, timestamps, epochs, labels = zip(*[load_nwb(str(f)) for f in files])
    # Merge data by simple concat
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    kin_mask = np.concatenate(kin_mask, axis=0)
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
    return binned, kin, kin_mask, trial_num, timestamps, epochs, labels[0]

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
        task: ExperimentalTask,
        **kwargs,
    ):
        assert cfg.bin_size_ms == 20, "FALCON data needs 20ms"
        # Load data
        binned, kin, kin_mask, trials, timestamps, epochs, labels = load_files([datapath])
        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        for trial_id in np.unique(trials):
            trial_spikes = binned[trials == trial_id]
            if len(trial_spikes) < 5:
                logger.warning(f"Skipping trial {trial_id} with only {len(trial_spikes)} bins")
                continue
            trial_vel = kin[trials == trial_id]
            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg, spike_bin_size_ms=cfg.bin_size_ms),
                DataKey.bhvr_vel: torch.tensor(trial_vel.copy()),
                DataKey.bhvr_mask: torch.tensor(kin_mask[trials == trial_id].copy()).unsqueeze(-1),
            }
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)