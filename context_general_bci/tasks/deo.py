r"""
    Deo Bimanual Data https://datadryad.org/stash/dataset/doi:10.5061/dryad.sn02v6xbb
    We restrict to only mixed
"""
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from einops import reduce, rearrange
import scipy.signal as signal

from context_general_bci.utils import loadmat
from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import (
    chop_vector,
    compress_vector,
    spike_times_to_dense,
    heuristic_sanitize_payload
)

# Extracted from the readme.txt
ol_blocks = {
    "t5_06_02_2021": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    "t5_06_04_2021": [2, 3, 4, 5, 6],
    "t5_06_23_2021": [6, 7, 8, 9],
    "t5_06_28_2021": [5, 6, 7, 8, 9],
    "t5_06_30_2021": [9, 10, 11, 12, 13],
    "t5_07_12_2021": [8, 9, 10, 11, 12],
    "t5_07_14_2021": [7, 8, 14, 15],
    "t5_10_11_2021": [7, 9, 19, 28],
    "t5_10_13_2021": [5, 7, 9, 27],
    # These are only unimanual OL
    # "t5_09_13_2021": [1, 2, 3, 4],
    # "t5_09_15_2021": [1, 2, 3, 5],
    # "t5_09_27_2021": [1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27],
    # "t5_09_29_2021": [1, 2, 3, 4, 5, 6, 7],
    # "t5_10_11_2021": [1, 4, 6, 8],
    # "t5_10_13_2021": [2, 4, 6, 8],
}

@ExperimentalTaskRegistry.register
class DeoLoader(ExperimentalTaskLoader):
    r"""
        Bimanual 2D cursor data. We only analyze open loop blocks.
    """
    name = ExperimentalTask.deo

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
        exp_task_cfg = cfg.deo
        payload = loadmat(datapath)
        cursor_pos = payload['Cp'].T # Already 20ms bless -> Tx4
        tx = payload['tx'].T # Already 20ms, -> Tx192
        all_bhvr = np.gradient(cursor_pos, axis=0)
        assert all_bhvr.shape[0] == tx.shape[0]
        # No data mask for now.

        # Restrict to open loop timesteps
        if datapath.stem in ol_blocks:
            ol_timesteps = np.isin(payload['blockNum'], ol_blocks[datapath.stem])
            all_bhvr = torch.as_tensor(all_bhvr[ol_timesteps])
            all_spikes = torch.as_tensor(tx[ol_timesteps])
        else:
            return pd.DataFrame()
        meta_payload = {}
        meta_payload['path'] = []
        assert exp_task_cfg.chop_size_ms, "Trialized proc not supported"
        all_bhvr = chop_vector(all_bhvr, exp_task_cfg.chop_size_ms, cfg.bin_size_ms).to(torch.float32)
        all_spikes = chop_vector(all_spikes, exp_task_cfg.chop_size_ms, cfg.bin_size_ms)
        for t in range(len(all_bhvr)):
            single_payload = {
                DataKey.spikes: create_spike_payload(all_spikes[t], context_arrays),
                DataKey.bhvr_vel: all_bhvr[t].clone(), # T x H
            }
            # Extremely hacky patch to match the preprocessing in NDT3, which erases a specific single trial
            # from the heuristic due to heuristic triggering after normalization, which NDT2 lacks.
            if t == 2967 and datapath.stem == "t5_06_02_2021":
                continue
            if not heuristic_sanitize_payload(single_payload):
                continue
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        print(f"{datapath}, Saved {len(meta_payload['path'])} trials")
        return pd.DataFrame(meta_payload)
