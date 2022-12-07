from typing import Dict, List, Any
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
from einops import rearrange

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors

from config import DataKey, DatasetConfig
from subjects import SubjectInfo
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
        phase='test'
    ):
        r"""
            Loader for motor tasks in Neural Latents Benchmark (NLB) dataset.
        """
        dataset = NWBDataset(datapath)
        dataset.resample(cfg.bin_size_ms)

        # Create suffix for group naming later
        # suffix = '' if (cfg.bin_size_ms == 5) else f'_{int(cfg.bin_size_ms)}'
        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_train_input_tensors(
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
        train_spikes_heldin = torch.tensor(train_spikes_heldin)
        # train_spikes_heldin = torch.tensor(train_spikes_heldin, dtype=torch.uint8)
        meta_payload = {}
        meta_payload['path'] = []

        for trial in range(train_spikes_heldin.shape[0]):
            single_payload = {
                DataKey.spikes: {
                    # TODO split into PMd and M1
                    subject.wrap_array(list(subject.arrays.keys())[0]): rearrange(train_spikes_heldin[trial], 't c -> t c 1').clone()
                },
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