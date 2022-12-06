from typing import Dict, Any
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from einops import rearrange

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors

from config import DataKey, MetaKey, DatasetConfig
from context_registry import context_registry
from subjects import SubjectArrayRegistry, SubjectName, ArrayID
from tasks.task_registry import ExperimentalTaskLoader, ExperimentalTaskRegistry
TrialNum = int
MetadataKey = str


# Core loading strategy pulled from https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb

class NLBLoader(ExperimentalTaskLoader):
    name = "nlb_base"

    def load(path: Path, cfg: DatasetConfig, cache_root: Path, phase='test'):
        r"""
            Loader for motor tasks in Neural Latents Benchmark (NLB) dataset.
        """
        dataset = NWBDataset(path)
        dataset.resample(cfg.bin_size_ms)

        # Create suffix for group naming later
        # suffix = '' if (cfg.bin_size_ms == 5) else f'_{int(cfg.bin_size_ms)}'
        context_info = context_registry.query_by_datapath(path)
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_train_input_tensors(
            dataset,
            dataset_name=context_info.alias,
            trial_split=train_split,
            save_file=False
        )
        import pdb;pdb.set_trace()

        # Show fields of returned dict
        # print(train_dict.keys())

        # Unpack data
        train_spikes_heldin = train_dict['train_spikes_heldin']
        # train_spikes_heldout = train_dict['train_spikes_heldout']

        # Print 3d array shape: trials x time x channel
        # print(train_spikes_heldin.shape)

        meta_payload = {}
        meta_payload['path'] = []
        for trial in range(train_spikes_heldin.shape[0]):
            single_payload = {
                DataKey.spikes: rearrange(train_spikes_heldin[trial], 't c -> t c 1'),
            }
            meta_payload['path'].append(cache_root / f"{trial}.pth")
            torch.save(single_payload, meta_payload['path'])
        return pd.DataFrame(meta_payload)

@ExperimentalTaskRegistry.register
class MazeLoader(NLBLoader):
    name = "maze"

@ExperimentalTaskRegistry.register
class RTTLoader(NLBLoader):
    name = "random_target_task"