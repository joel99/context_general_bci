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

from context_general_bci.config import DataKey, DatasetConfig, FalconConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

# Load nwb file
def _bin_units_old( # Used for M1/H1 runs
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


# Load nwb file
def bin_units(
        units: pd.DataFrame,
        bin_size_s: float = 0.02,
        bin_timestamps: Optional[np.ndarray] = None,
        is_timestamp_bin_start: bool = False,
    ) -> np.ndarray:
    r"""
        Bin spikes given by an nwb units dataframe.
        There is one bin per input timestamp. If timestamps are not provided, they are inferred from the spike times.
        Timestamps are ideally provided spaced bin_size_s apart.
        If not:
        - if consecutive interval is greater than bin_size_s, then only the proximal bin_size_s interval is used.
        - if consecutive interval is less than bin_size_s, spikes will be binned in the provided interval (i.e. those bins will be smaller than bin_size_s apart).
            - Not outputting repeated spikes mainly because the implementation would be more complex (no single call to np.histogram)
        Args:
            units: df with only index (spike index) and spike times (list of times in seconds). From nwb.units.
            bin_size_s: size of each bin to output in seconds.
            bin_timestamps: array of timestamps indicating bin time in seconds.
            is_timestamp_bin_start: if True, the bin is considered to start at the timestamp, otherwise it ends at the timestamp.
        Returns:
            array of spike counts per bin, per unit. Shape is (bins x units).
    """
    # Make even bins
    if bin_timestamps is None:
        end_time = units.spike_times.apply(lambda s: max(s) if len(s) else 0).max() + bin_size_s
        bin_end_timestamps = np.arange(0, end_time, bin_size_s)
        bin_mask = np.ones(len(bin_end_timestamps), dtype=bool)
    else:
        if is_timestamp_bin_start:
            bin_end_timestamps = bin_timestamps + bin_size_s
        else:
            bin_end_timestamps = bin_timestamps
        # Check contiguous else force cropping for even bins
        gaps = np.diff(bin_end_timestamps)
        if (gaps <= 0).any():
            raise ValueError("bin_end_timestamps must be monotonically increasing.")
        if not np.allclose(gaps, bin_size_s):
            print(f"Warning: Input timestamps not spaced like requested {bin_size_s}. Outputting proximal bin spikes.")
            # Adjust bin_end_timestamps to include bins at the end of discontinuities
            new_bin_ends = [bin_end_timestamps[0]]
            bin_mask = [True] # bool, True if bin ending at this timepoint should be included post mask (not padding)
            for i, gap in enumerate(gaps):
                if not np.isclose(gap, bin_size_s) and gap > bin_size_s:
                    cur_bin_end = bin_end_timestamps[i+1]
                    new_bin_ends.extend([cur_bin_end - bin_size_s, cur_bin_end])
                    bin_mask.extend([False, True])
                else:                        
                    new_bin_ends.append(bin_end_timestamps[i+1])
                    bin_mask.append(True)
            bin_end_timestamps = np.array(new_bin_ends)
            bin_mask = np.array(bin_mask)
        else:
            bin_mask = np.ones(len(bin_end_timestamps), dtype=bool)

    # Make spikes
    spike_arr = np.zeros((bin_mask.sum(), len(units)), dtype=np.uint8)
    bin_edges = np.concatenate([np.array([bin_end_timestamps[0] - bin_size_s]), bin_end_timestamps])
    for idx, (_, unit) in enumerate(units.iterrows()):
        spike_cnt, _ = np.histogram(unit.spike_times, bins=bin_edges)
        if bin_mask is not None:
            spike_cnt = spike_cnt[bin_mask]
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
            bin_units(units, bin_timestamps=timestamps),
            kin,
            kin_mask,
            trial_num,
            # timestamps,
            # epochs,
            # labels
        )

def load_files_h1(files: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binned, kin, kin_mask, trial_num = zip(*[load_nwb_h1(str(f)) for f in files])
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
            binned_units = bin_units(units, bin_size_s=0.02, bin_timestamps=emg_timestamps)

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

def load_files_m2(files: List):
    out_neural = []
    out_cov = []
    out_mask = []
    out_trial = []
    for fn in files:
        with NWBHDF5IO(fn, 'r') as io:
            nwbfile = io.read()
            units = nwbfile.units.to_dataframe()
            vel_container = nwbfile.acquisition['finger_vel']
            labels = [ts for ts in vel_container.time_series]
            vel_data = []
            vel_timestamps = None
            for ts in labels:
                ts_data = vel_container.get_timeseries(ts)
                vel_data.append(ts_data.data[:])
                vel_timestamps = ts_data.timestamps[:]
            vel_data = np.vstack(vel_data).T
            # TODO check bin timestamp appropriate?
            binned_units = bin_units(units, bin_size_s=0.02, bin_timestamps=vel_timestamps, is_timestamp_bin_start=True)

            eval_mask = nwbfile.acquisition['eval_mask'].data[:].astype(bool)

            trial_change = np.zeros(vel_timestamps.shape[0], dtype=bool)
            trial_info = nwbfile.trials.to_dataframe().reset_index()
            switch_inds = np.searchsorted(vel_timestamps, trial_info.start_time)
            trial_change[switch_inds] = True
            trial_dense = np.cumsum(trial_change)
            out_neural.append(binned_units)
            out_cov.append(vel_data)
            out_mask.append(eval_mask)
            out_trial.append(trial_dense)
    binned_units = np.concatenate(out_neural, axis=0)
    cov_data = np.concatenate(out_cov, axis=0)
    eval_mask = np.concatenate(out_mask, axis=0)
    trial_dense = np.concatenate(out_trial, axis=0)
    return binned_units, cov_data, eval_mask, trial_dense

class HandwritingTokenizer:
    VOCAB = ["'", ',', '>', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']
    
    @staticmethod
    def tokenize(text: str) -> np.ndarray:
        return np.array([HandwritingTokenizer.VOCAB.index(c) for c in text], dtype=np.int32) + 1 # 0 is CTC blank
    
    @staticmethod
    def detokenize(tokens: np.ndarray) -> str:
        return ''.join([HandwritingTokenizer.VOCAB[t - 1] for t in tokens]) # 0 is blank

def load_files_h2(files: List):
    out_neural = []
    out_cov = []
    out_mask = []
    out_trial = []
    for fn in files:
        with NWBHDF5IO(fn, 'r') as io:
            nwbfile = io.read()
            binned_spikes = nwbfile.acquisition['binned_spikes'].data[()]
            time = nwbfile.acquisition['binned_spikes'].timestamps[()]
            eval_mask = nwbfile.acquisition['eval_mask'].data[()].astype(bool)
            trial_info = (
                nwbfile.trials.to_dataframe()
                .reset_index()
            )
            targets = []
            for _, row in trial_info.iterrows():
                # targets.append(np.array([ord(c) for c in row.cue], dtype=np.int32))
                targets.append(HandwritingTokenizer.tokenize(row.cue))
            trial_change = np.concatenate([np.diff(time) > 0.021, [True]])
            trial_dense = np.cumsum(trial_change)
            out_neural.append(binned_spikes)
            out_cov.append(targets)
            out_mask.append(eval_mask)
            out_trial.append(trial_dense)
    # Do not concatenate - trialized data
    return out_neural, out_cov, out_mask, out_trial

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
        if task != ExperimentalTask.falcon_h2:
            assert cfg.bin_size_ms == 20, "FALCON data needs 20ms"
        # Load data
        if task == ExperimentalTask.falcon_h1:
            binned, kin, kin_mask, trials = load_files_h1([datapath])
        elif task == ExperimentalTask.falcon_h2:
            binned, kin, kin_mask, trials = load_files_h2([datapath])
            binned = binned[0][:-1] # crop last bin
            kin = kin[0]
            kin_mask = kin_mask[0][:-1]
            trials = trials[0][:-1]
        elif task == ExperimentalTask.falcon_m1:
            binned, kin, kin_mask, trials = load_files_m1([datapath])
        elif task == ExperimentalTask.falcon_m2:
            binned, kin, kin_mask, trials = load_files_m2([datapath])
        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        exp_task_cfg: FalconConfig = getattr(cfg, task.value)
        if task == ExperimentalTask.falcon_h2:
            assert exp_task_cfg.respect_trial_boundaries, "Falcon H2 data must respect trial boundaries"
        if exp_task_cfg.respect_trial_boundaries:
            for trial_id in np.unique(trials):
                trial_spikes = binned[trials == trial_id]
                if len(trial_spikes) < 5:
                    logger.warning(f"Skipping trial {trial_id} with only {len(trial_spikes)} bins")
                    continue
                if task == ExperimentalTask.falcon_h2:
                    trial_vel = kin[trial_id]
                    trial_vel = torch.tensor(trial_vel[:, None], dtype=int) # add a spatial dimension
                else:
                    trial_vel = kin[trials == trial_id]
                    trial_vel = torch.tensor(trial_vel.copy(), dtype=torch.float32)
                single_payload = {
                    DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg, spike_bin_size_ms=cfg.bin_size_ms),
                    DataKey.bhvr_vel: trial_vel, # need float, not double
                    DataKey.bhvr_mask: torch.tensor(kin_mask[trials == trial_id].copy()).unsqueeze(-1),
                }
                single_path = cache_root / f'{trial_id}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        else:
            chop_size_bins = exp_task_cfg.chop_size_ms // cfg.bin_size_ms
            for i in range(0, binned.shape[0], chop_size_bins):
                trial_spikes = binned[i:i+chop_size_bins]
                if len(trial_spikes) < 5:
                    logger.warning(f"Skipping trial {i} with only {len(trial_spikes)} bins")
                    continue
                trial_vel = kin[i:i+chop_size_bins]
                single_payload = {
                    DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg, spike_bin_size_ms=cfg.bin_size_ms),
                    DataKey.bhvr_vel: torch.tensor(trial_vel.copy(), dtype=torch.float32), # need float, not double
                    DataKey.bhvr_mask: torch.tensor(kin_mask[i:i+chop_size_bins].copy()).unsqueeze(-1),
                }
                single_path = cache_root / f'{i}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)

ExperimentalTaskRegistry.register_manual(ExperimentalTask.falcon_h2, FalconLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.falcon_m1, FalconLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.falcon_m2, FalconLoader)