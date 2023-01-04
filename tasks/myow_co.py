#%%
r"""
A handful of clearly sorted ~700 trials from MYOW
https://github.com/nerdslab/myow-neuro
- Adapted from myow/data/monkey_neural_dataset.py
"""
from typing import List
from pathlib import Path
import os
import pickle
import logging

import numpy as np
from tqdm import tqdm

from config import DataKey, DatasetConfig
from subjects import SubjectInfo, SubjectArrayRegistry
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

from utils import loadmat

logger = logging.getLogger(__name__)

@ExperimentalTaskRegistry.register
class DyerCODataset(ExperimentalTaskLoader):
    name = ExperimentalTask.dyer_co
    FILENAMES = {
        ('mihi', 1): 'full-mihi-03032014',
        ('mihi', 2): 'full-mihi-03062014',
        ('chewie', 1): 'full-chewie-10032013',
        ('chewie', 2): 'full-chewie-12192013',
    }

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
    # def __init__(self, root, primate='mihi', day=1, split='train', train_split=0.8, val_split=0.1,
                #  binning_period=0.005, velocity_threshold=5.):

        self.root = root
        assert primate in ['mihi', 'chewie']
        assert day in [1, 2]
        self.primate = primate
        self.day = day

        # get path to data
        self.filename = self.FILENAMES[(self.primate, day)]
        self.raw_path = os.path.join(self.root, 'dyer_co', '%s.mat') % self.filename
        # self.raw_path = os.path.join(self.root, 'raw', '%s.mat') % self.filename
        self.processed_path = os.path.join(self.root, 'preprocessed/dyer_co/{}-bin{:.2f}-vel_thresh{:.2f}.pkl'.format(
        # self.processed_path = os.path.join(self.root, 'processed/{}-bin{:.2f}-vel_thresh{:.2f}.pkl'.format(
            self.filename, binning_period, velocity_threshold))

        # get pre-processing parameters
        self.binning_period = binning_period
        self.velocity_threshold = velocity_threshold

        # load processed data or process data
        if not os.path.exists(self.processed_path):
            data = self._process()
        import pdb;pdb.set_trace()
        # else:
            # data_train_test = self._load_processed_data()

        # # train/val split
        # assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        # self.split = split
        # self.train_split = train_split
        # self.val_split = val_split

        # # split data
        # self.data = self._split_train_test(data_train_test, split=split)

    def __repr__(self):
        return '{}(primate={}, day={}, split={})'.format(self.__class__.__name__, self.primate, self.day, self.split)

    def _process(self):
        logging.info('Preparing dataset: Binning data.')
        # load data
        mat_dict = loadmat(self.raw_path)

        # bin data
        data = self._bin_data(mat_dict)
        # convert to graphs
        data = self._convert_to_graphs(data)

        self._save_processed_data(data)
        return data

    def _bin_data(self, mat_dict):
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

        data_list = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                     'force': [], 'labels': [], 'sequence': []}
        for trial_id in tqdm(range(num_trials)):
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            # grids= minT:(delT-TO):(maxT-delT);
            grid = np.arange(min_T, max_T + self.binning_period, self.binning_period)
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
                    neurons_binned[k, i] = np.sum(bin_mask) / self.binning_period

            # filter velocity
            mask = np.linalg.norm(vel_binned, 2, axis=1) > self.velocity_threshold

            data_list['firing_rates'].append(neurons_binned[mask])
            # data_list['position'].append(pos_binned[mask])
            data_list['velocity'].append(vel_binned[mask])
            data_list['acceleration'].append(acc_binned[mask])
            data_list['force'].append(force_binned[mask])
            data_list['labels'].append(targets_binned[mask])
            # data_list['sequence'].append(id_binned[mask])

            single_payload = {
                DataKey.spikes: spike_payload,
                DataKey.bhvr_vel: bhvr_vars[DataKey.bhvr_vel][t].clone(),
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
            import pdb;pdb.set_trace()

        return data_list

    def _save_processed_data(self, data):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        with open(self.processed_path, 'wb') as output:
            pickle.dump({'data': data}, output)
        logging.info('Processed data was saved to {}.'.format(self.processed_path))

test_data = DyerCODataset(
    root='data',
    primate='mihi',
    day=1,
)