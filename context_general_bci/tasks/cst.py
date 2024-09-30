#%%

from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import scipy.signal as signal

import logging
logger = logging.getLogger(__name__)

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector,
    compress_vector,
    heuristic_sanitize,
)
from context_general_bci.utils import loadmat

@ExperimentalTaskRegistry.register
class CSTLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.cst
    r"""
    Batista Chowdhury colab - Critical stabliity task
    """
    @staticmethod
    def reduce_condition(spikes, bhvr_vars, target_condition):
        new_spikes = []
        new_bhvr_vars = {k: [] for k in bhvr_vars.keys()}
        for i, c in enumerate(bhvr_vars['condition']):
            if c == target_condition:
                new_spikes.append(spikes[i])
                for k in bhvr_vars.keys():
                    new_bhvr_vars[k].append(bhvr_vars[k][i])
        return new_spikes, new_bhvr_vars

    @staticmethod
    def load_raw(datapath: Path, cfg: DatasetConfig, context_arrays: List[str]):
        r"""
            `.mat` file
            Spike/bhvr loaded at 1khz (1ms bin size), behavior loaded at target bin size (20ms). Required to round up spikes to match.
        """
        # Hacky patch to determine the right arrays to use
        payload = loadmat(datapath)['trial_data'] # has keys, each a list of length Trials (i.e. not continuous data)
        bin_size = payload['bin_size']
        assert all(b == 0.001 for b in bin_size), "All bin sizes must be the same and expected a 1ms"
        # impt keys: M1 spikes (sorted, 2x96 max), hand pos, task (CST or CO)
        spike_arr = payload['M1_spikes']
        pos_arr = payload['hand_pos']
        spikes = []
        positions = []
        assert [t in ['CST', 'CO'] for t in payload['task']], "Task must be CST or CO"
        for i,t in enumerate(payload['task']):
            if not isinstance(pos_arr[i], np.ndarray): # Something odd
                spikes.append(None)
                positions.append(None)
                continue
            time_mask = np.zeros(spike_arr[i].shape[0], dtype=bool)
            if t == 'CST':
                if not np.isnan(payload['idx_cstStartTime'][i]):
                    time_mask[int(payload['idx_cstStartTime'][i]):int(payload['idx_cstEndTime'][i])] = True
                else:
                    # Something is weird about the trial, discard
                    pass
            else:
                # print(time_mask.shape, payload['idx_startTime'][i], payload['idx_otHoldTime'][i])
                if np.isnan(payload['idx_otHoldTime'][i]).any():
                    pass
                else:
                    time_mask[:int(payload['idx_otHoldTime'][i])] = True
            spikes.append(spike_arr[i][time_mask])
            positions.append(pos_arr[i][time_mask][:, :1]) # Only use X
        conditions = [0 if t == 'CO' else 1 for t in payload['task']] # 1 for CST
        # filter out fails
        spikes, positions, conditions = zip(*[(s, p, c) for s, p, c in zip(spikes, positions, conditions) if s is not None])
        bhvr_vars = {
            'pos': positions,
            'condition': conditions
        }
        return spikes, bhvr_vars, context_arrays

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
    ):
        exp_task_cfg = cfg.cst
        meta_payload = {}
        if datapath.suffix == '.pth': # from `split_eval`
            payload = torch.load(datapath)
            spike_arr = payload['spikes']
            bhvr_vars = {k: payload[k] for k in payload.keys() if k != 'spikes'}
            bhvr_vars['condition'] = [1 for _ in spike_arr]
        else:
            spike_arr, bhvr_vars, context_arrays = cls.load_raw(datapath, cfg, context_arrays)
        def resample(data): # Updated 9/10/23: Previous resample produces an undesirable strong artifact at timestep 0. This hopefully removes that and thus also removes outliers.
            return torch.tensor(
                signal.resample_poly(data, 1, cfg.bin_size_ms, padtype='line')
            )
        # Bhvr is not ragged, but padded with NaNs. First determine the true length of data, and convert to ragged.
        # Spotchecking, nans only appear contiguously at end of trial
        trialized_spikes = []
        trialized_bhvr = []
        trialized_cond = []
        for i, k in enumerate(bhvr_vars['pos']): # Trial x Time x 3
            if k.ndim == 1:
                k = k[:, None] # add an extra dim - we might've squeezed it out
            if np.isnan(k).any():
                first_nan = np.where(np.isnan(k))[0][0]
                cur_pos = k[:first_nan]
                cur_spikes = spike_arr[i][:first_nan]
            else:
                cur_pos = k
                cur_spikes = spike_arr[i]
            if not heuristic_sanitize(cur_spikes, cur_pos):
                continue
            elif cur_pos.shape[0] > 8000:
                raise ValueError(f"Trial {i} has strange length ({cur_pos.shape}), need to scrub data for length filters")
            trialized_spikes.append(cur_spikes)
            downsample_bhvr = resample(cur_pos)
            downsample_vel = np.gradient(downsample_bhvr, axis=0)
            trialized_bhvr.append(downsample_vel)
            trialized_cond.append(bhvr_vars['condition'][i])
        trialized_spikes = [compress_vector(s, 0, cfg.bin_size_ms) for s in trialized_spikes]
        if len(trialized_spikes) == 0:
            # null dataset, a bit concerning
            # will likely flag downstream if important eval
            return pd.DataFrame(meta_payload)
        bhvr_vars[DataKey.bhvr_vel] = trialized_bhvr

        meta_payload['path'] = []
        for t in range(len(trialized_spikes)):
            trial_spikes = trialized_spikes[t]
            trial_bhvr = bhvr_vars[DataKey.bhvr_vel][t]
            if trial_bhvr.shape[0] > trial_spikes.shape[0] and trial_bhvr.shape[0] - trial_spikes.shape[0] <= 2:
                trial_bhvr = trial_bhvr[:trial_spikes.shape[0]]
            if trial_spikes.shape[0] != trial_bhvr.shape[0]:
                logger.warning(f"Trial {t} has mismatched bhvr and spikes")
                continue
            if exp_task_cfg.chop_size_ms:
                trial_spikes = chop_vector(torch.as_tensor(trial_spikes[..., 0]), exp_task_cfg.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1)
                trial_bhvr = chop_vector(torch.as_tensor(trial_bhvr), exp_task_cfg.chop_size_ms, cfg.bin_size_ms)
                for i in range(len(trial_spikes)):
                    single_payload = {
                        DataKey.spikes: create_spike_payload(trial_spikes[i], context_arrays),
                        DataKey.bhvr_vel: trial_bhvr[i].to(torch.float32),
                    }
                    single_path = cache_root / f'{t}_{i}.pth'
                    meta_payload['path'].append(single_path)
                    torch.save(single_payload, single_path)
            else:
                single_payload = {
                    DataKey.spikes: create_spike_payload(trial_spikes, context_arrays),
                    DataKey.bhvr_vel: trial_bhvr,
                }

                single_path = cache_root / f'{t}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
