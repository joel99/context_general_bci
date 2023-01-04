#%%
r"""
Gallego CO release (10/29/22)
https://doi.org/10.5061/dryad.xd2547dkt

```
In these experiments, monkeys controlled a cursor on a screen using a two-link, planar manipulandum. Monkeys performed a simple center-out task to one of the eight possible targets, after a variable delayed period. During this reaching task, we tracked the endpoint position of the hand using sensors on the manipulandum. In addition to the behavioral data, we collected neural data from one or two of these areas using Blackrock Utah multielectrode arrays, yielding ~100 to ~200 channels of extracellular recordings per monkey. Recordings from these channels were thresholded online to detect spikes, which were sorted offline into putative single units.
```

Data was pulled by GUI download + transfer to cluster. (no CLI)
"""
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DataKey, DatasetConfig
from subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

from utils import loadmat

import logging

logger = logging.getLogger(__name__)

@ExperimentalTaskRegistry.register
class GallegoCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.gallego_co

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
        raise NotImplementedError
        mat_dict = loadmat(datapath)

        # load matrix
        trialtable = mat_dict['trial_table']
        neurons = mat_dict['out_struct']['units']
        # pos = np.array(mat_dict['out_struct']['pos'])
        vel = np.array(mat_dict['out_struct']['vel'])
        acc = np.array(mat_dict['out_struct']['acc'])
        force = np.array(mat_dict['out_struct']['force'])
        time = vel[:, 0]

        num_neurons = len(neurons)
        num_trials = trialtable.shape[0]

        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        # data_list = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                    #  'force': [], 'labels': [], 'sequence': []}
        for trial_id in range(num_trials):
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            binning_period = cfg.bin_size_ms / 1000
            grid = np.arange(min_T, max_T + binning_period, binning_period)
            grids = grid[:-1]
            gride = grid[1:]
            num_bins = len(grids)

            neurons_binned = np.zeros((num_bins, num_neurons))
            # pos_binned = np.zeros((num_bins, 2))
            vel_binned = np.zeros((num_bins, 2))
            acc_binned = np.zeros((num_bins, 2))
            force_binned = np.zeros((num_bins, 2))
            targets_binned = np.zeros((num_bins,))
            # id_binned = np.arange(num_bins)

            for k in range(num_bins):
                bin_mask = (time >= grids[k]) & (time <= gride[k])
                # if len(pos) > 0:
                    # pos_binned[k, :] = np.mean(pos[bin_mask, 1:], axis=0)
                vel_binned[k, :] = np.mean(vel[bin_mask, 1:], axis=0)
                if len(acc):
                    acc_binned[k, :] = np.mean(acc[bin_mask, 1:], axis=0)
                if len(force) > 0:
                    force_binned[k, :] = np.mean(force[bin_mask, 1:], axis=0)
                targets_binned[k] = trialtable[trial_id, 1]

            for i in range(num_neurons):
                for k in range(num_bins):
                    spike_times = neurons[i]['ts']
                    bin_mask = (spike_times >= grids[k]) & (spike_times <= gride[k])
                    neurons_binned[k, i] = np.sum(bin_mask) / binning_period

            # filter velocity
            mask = np.linalg.norm(vel_binned, 2, axis=1) > cfg.dyer_co.velocity_threshold
            # data_list['firing_rates'].append(neurons_binned[mask])
            # data_list['position'].append(pos_binned[mask])
            # data_list['velocity'].append(vel_binned[mask])
            # data_list['acceleration'].append(acc_binned[mask])
            # data_list['force'].append(force_binned[mask])
            # data_list['labels'].append(targets_binned[mask])
            # data_list['sequence'].append(id_binned[mask])
            single_payload = {
                DataKey.spikes: create_spike_payload(torch.tensor(neurons_binned[mask]), arrays_to_use),
                DataKey.bhvr_vel: torch.tensor(vel_binned[mask]),
                DataKey.bhvr_acc: torch.tensor(acc_binned[mask]),
                DataKey.bhvr_force: torch.tensor(force_binned[mask]),
            }
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
