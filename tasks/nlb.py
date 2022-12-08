from typing import Dict, List, Any
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from einops import rearrange

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, PARAMS, _prep_mask, make_stacked_array

from config import DataKey, DatasetConfig
from subjects import SubjectInfo, SubjectName, SubjectArrayRegistry
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
TrialNum = int
MetadataKey = str


# Core loading strategy pulled from https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb

class NLBLoader(ExperimentalTaskLoader):
    name = "nlb_base"

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        phase='test',
        dataset_cls=NWBDataset,
        make_tensor_fn=make_train_input_tensors
    ):
        r"""
            Loader for motor tasks in Neural Latents Benchmark (NLB) dataset.
        """
        dataset = dataset_cls(datapath)
        dataset.resample(cfg.bin_size_ms)

        # Create suffix for group naming later
        # suffix = '' if (cfg.bin_size_ms == 5) else f'_{int(cfg.bin_size_ms)}'
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_tensor_fn(
            dataset,
            dataset_name=dataset_alias,
            trial_split=train_split,
            save_file=False
        )

        # Show fields of returned dict
        # print(train_dict.keys())

        # Unpack data
        train_spikes_heldin = train_dict['train_spikes_heldin']
        # train_spikes_heldout = train_dict['train_spikes_heldout']

        # Print 3d array shape: trials x time x channel
        # print(train_spikes_heldin.shape)
        # train_spikes_heldin = torch.tensor(train_spikes_heldin)
        train_spikes_heldin = torch.tensor(train_spikes_heldin, dtype=torch.uint8)
        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays

        for trial in range(train_spikes_heldin.shape[0]):
            spikes = rearrange(train_spikes_heldin[trial], 't c -> t c 1')
            spike_payload = {}
            for a in arrays_to_use:
                array = SubjectArrayRegistry.query_by_array(a)
                if array.is_exact:
                    array = SubjectArrayRegistry.query_by_array_geometric(a) # get some typing
                    spike_payload[a] = spikes[:, array.as_indices()].clone()
                else:
                    assert len(arrays_to_use) == 1, "Can't use multiple arrays with non-exact arrays"
                    spike_payload[a] = spikes.clone()
            single_payload = {
                DataKey.spikes: spike_payload
            }
            single_path = cache_root / f"{trial}.pth"
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)

@ExperimentalTaskRegistry.register
class MazeLoader(NLBLoader):
    name = ExperimentalTask.maze

@ExperimentalTaskRegistry.register
class RTTLoader(NLBLoader):
    name = ExperimentalTask.rtt


# === Overrides for Churchland dataset ===
class NWBDatasetChurchland(NWBDataset):
    def __init__(self, *args, **kwargs):
        kwargs['split_heldout'] = False
        kwargs['skip_fields'] = [
            'Position_Cursor',
            'Position_Eye',
            'Position_Hand',
            'Processed_A001',
            'Processed_A002',
            'Processed_B001',
            'Processed_B002',
        ]
        # Note - currently these fields are dropped due to a slight timing mismatch.
        # If you want them back, you'll need to reduce precision in NWBDataset.load() from 6 digits to 3 digits (which I think is fine)
        # But we currently don't need
        super().__init__(*args, **kwargs)
        self.trial_info = self.trial_info.rename({ # match NLB naming
            'move_begins_time': 'move_onset_time',
            'task_success': 'success',
            'target_presentation_time': 'target_on_time',
            'reaction_time': 'rt',
        }, axis=1)

        self.trial_info['split'] = 'train'

def make_input_tensors_simple(dataset, mock_dataset='mc_maze', trial_split=['train'], **kwargs):
    # See `make_train_input_tensors` for documentation
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"

    # Fetch and update params
    params = PARAMS[mock_dataset].copy()
    # unpack params
    spk_field = params['spk_field']
    # hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()

    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Make output spiking arrays and put into data_dict
    train_dict = make_stacked_array(dataset, [spk_field], make_params, trial_mask)
    return {
        'train_spikes_heldin': train_dict[spk_field]
    }

@ExperimentalTaskRegistry.register
class ChurchlandMazeLoader(NLBLoader):
    name = ExperimentalTask.churchland_maze

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        phase='test',
        dataset_cls=NWBDatasetChurchland,
        make_tensor_fn=make_input_tensors_simple
    ):
        return super().load(
            datapath,
            cfg,
            cache_root,
            subject,
            context_arrays,
            dataset_alias,
            phase,
            dataset_cls,
            make_tensor_fn
        )