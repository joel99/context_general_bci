r"""
- Dataloaders adapted from falcon_challenge.dataloaders
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

def load_nwb_h1(fn: str):
    r"""
        Load NWB for Human Motor ARAT dataset. Kinematic timestamps are provided at 50Hz.
    """
    with NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        kin = nwbfile.acquisition['OpenLoopKinematicsVelocity'].data[:].astype(dtype=np.float32)
        kin_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)
        trial_num = nwbfile.acquisition["TrialNum"].data[:]
        timestamps = nwbfile.acquisition['OpenLoopKinematics'].offset + np.arange(kin.shape[0]) * nwbfile.acquisition['OpenLoopKinematics'].rate
        # labels = [l.strip() for l in nwbfile.acquisition['OpenLoopKinematics'].description.split(',')]
        # epochs = nwbfile.epochs.to_dataframe()
        return (
            bin_units(units, bin_end_timestamps=timestamps),
            kin,
            kin_mask,
            trial_num,
            # timestamps,
            # epochs,
            # labels
        )

def load_files_h1(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binned, kin, kin_mask, trial_num, timestamps = zip(*[load_nwb_h1(str(f)) for f in files])
    # Merge data by simple concat
    binned = np.concatenate(binned, axis=0)
    kin = np.concatenate(kin, axis=0)
    kin_mask = np.concatenate(kin_mask, axis=0)
    trial_num = np.concatenate(trial_num, axis=0)

    # Offset timestamps and epochs to be continuous across multiple datasets
    # all_timestamps = [timestamps[0]]
    # for idx, current_times in enumerate(timestamps[1:]):
        # epochs[idx]['start_time'] += all_timestamps[-1][-1] + 0.01 # 1 bin
        # epochs[idx]['stop_time'] += all_timestamps[-1][-1] + 0.01 # 1 bin
        # all_timestamps.append(current_times + all_timestamps[-1][-1] + 0.01)
    # timestamps = np.concatenate(all_timestamps, axis=0)
    # epochs = pd.concat(epochs, axis=0)
    # for l in labels[1:]:
        # assert l == labels[0]
    return binned, kin, kin_mask, trial_num, # timestamps # , labels[0]

def load_files_m1(files: List):
    out_neural = []
    out_cov = []
    out_mask = []
    out_trial = []
    for fn in files:
        with NWBHDF5IO(fn, 'r') as io:
            nwbfile = io.read()
            units = nwbfile.units.to_dataframe()
            raw_emg = nwbfile.acquisition['preprocessed_emg']
            muscles = [ts for ts in raw_emg.time_series]
            emg_data = []
            emg_timestamps = []
            for m in muscles:
                mdata = raw_emg.get_timeseries(m)
                data = mdata.data[:]
                timestamps = mdata.timestamps[:]
                emg_data.append(data)
                emg_timestamps.append(timestamps)
            emg_data = np.vstack(emg_data).T
            emg_timestamps = emg_timestamps[0]
            binned_units = bin_units(units, bin_size_s=0.02, bin_end_timestamps=emg_timestamps)

            eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)

            trial_info = (
                nwbfile.trials.to_dataframe()
                .reset_index()
                .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
            )
            switch_inds = np.searchsorted(emg_timestamps, trial_info.start_time)
            trial_change = np.zeros(emg_timestamps.shape[0], dtype=bool)
            trial_change[switch_inds] = True

            trial_dense = np.cumsum(trial_change)
            out_neural.append(binned_units)
            out_cov.append(emg_data)
            out_mask.append(eval_mask)
            out_trial.append(trial_dense)
    binned_units = np.concatenate(out_neural, axis=0)
    emg_data = np.concatenate(out_cov, axis=0)
    eval_mask = np.concatenate(out_mask, axis=0)
    trial_dense = np.concatenate(out_trial, axis=0)
    return binned_units, emg_data, eval_mask, trial_dense

@ExperimentalTaskRegistry.register
class FalconLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.falcon_h1

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
        if task == ExperimentalTask.falcon_h1:
            binned, kin, kin_mask, trials, _ = load_files_h1([datapath])
        elif task == ExperimentalTask.falcon_h2:
            raise NotImplementedError("Falcon H2 not implemented")
        elif task == ExperimentalTask.falcon_m1:
            binned, kin, kin_mask, trials = load_files_m1([datapath])
        elif task == ExperimentalTask.falcon_m2:
            raise NotImplementedError("Falcon M2 not implemented")
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
                DataKey.bhvr_vel: torch.tensor(trial_vel.copy(), dtype=torch.float32), # need float, not double
                DataKey.bhvr_mask: torch.tensor(kin_mask[trials == trial_id].copy()).unsqueeze(-1),
            }
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)

ExperimentalTaskRegistry.register_manual(ExperimentalTask.falcon_h2, FalconLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.falcon_m1, FalconLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.falcon_m2, FalconLoader)