from typing import Callable, Self, List, Any, Optional, Dict
import copy
from pathlib import Path
import re
import itertools
from datetime import datetime
from dataclasses import dataclass

import logging
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

from config import DatasetConfig, MetaKey, DataKey
from array_registry import subject_array_registry
from context_registry import context_registry

from tasks.passive_icms import infer_stim_parameters, icms_loader

r"""
    Stores range of contexts provided by a dataset.
    Data will serve attributes as index of provided contexts.
    The model should probably unique parameters for each context (JY thinks they should be embeddings).
    - `subject` context will _determine_ data shape.
    - TODO we need to figure out a multi-shape strategy.
    1. Simple is to get a linear layer per subject.
    2. Odd but maybe workable is to pad to some max length (e.g. 128, current generation Utah array).
    3. Stretch; too much of a separate research endeavor -- use Set networks to learn a subject-based cross-attention operator to downproject.
    These contexts are not independent. In fact, they're almost hierarchical.
    Subject -> Array -> Session, Task -> session.
    TODO think about whether Task should be here. Right now we're not attempting multitask. But multitask pretraining might be reasonable.
"""
@dataclass
class ContextAttrs:
    subject = Optional[List[str]]
    array = Optional[List[str]] # should be prefixed with subject # TODO (low priority) implement either 2 or 3
    session = Optional[List[str]] # should uniquely identify
    task = Optional[List[str]] # not to be confused with

@dataclass
class DataAttrs:
    bin_size_ms: int
    context: ContextAttrs


r"""
    A heterogenuous dataset needs some way of figuring out how to load data from different sources.
    We define loaders per data source, and will route using a data registry.
    This is not really designed with loading diverse data in the same instance; it's for doing so across instances/runs.
    Loader is responsible for preprocessing as well.
"""
SessionLoader = Callable[[Path, DatasetConfig, Path], pd.DataFrame]
def get_loader(path: Path) -> SessionLoader:
    if 'passive_icms' in path:
        return icms_loader
    if path.is_dir():
        assert False, "no directory handler yet"
    if path.suffix == '.nwb': # e.g. nlb
        assert False, "no NWB handler yet"
    if path.suffix == '.pth':
        assert False, "no pth handler yet"

class SpikingDataset(Dataset):
    r"""
        Generic container for spiking data from intracortical microelectrode recordings.
        Intended to wrap multiple (large) datasets, hence stores time series in disk, not memory.
        In order to be schema agnostic, we'll maintain metadata in a pandas dataframe and larger data (time series) will be stored in a file, per trial.
"        Some training modes may open a file and not use all info in said file, but if this turns into an issue we can revisit and split the files as well.

        Design considerations:
        - Try to be relatively agnostic (needs to deal with QuickLogger + NWB)
        # ? Will it be important to have a preprocessed cache? If trials are exploded, and we have distinct padding requirements, we might need per-trial processing. We might just store exploded values after preprocessing. But then, what if we need to update preprocessing?
        - Then we need to re-process + update exploded values. Simple as that.


        TODO implement. Should span several task settings.
        - ICMS
        - Motor cortex data

        Other notes:
        - Can we "mixin" time-varying data, or maybe simpler to just be a separate codepath in this class.
    """
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg
        assert DataKey.spikes in cfg.data_keys, "Must have spikes"

        meta_df = []
        for d in self.cfg.datasets:
            meta_df.append(self.load_single_session(d))
        self.meta_df = pd.concat(meta_df).reset_index()

        self.context_index = None
        self.subsetted = False

    @staticmethod
    def preprocess_path(cls, cfg: DatasetConfig, session_path: Path) -> Path:
        # TODO assert some sort of versioning system for preprocessing
        return cfg.root_dir / cfg.preprocess_suffix / session_path.name

    def validate_meta(self, meta_df: pd.DataFrame):
        assert all([k in meta_df.columns for k in self.cfg.meta_keys])
        if MetaKey.subject in self.cfg.meta_keys:
            unique_subjects = meta_df[MetaKey.subject].unique()
            for s in unique_subjects:
                assert subject_array_registry.query_by_subject(s) is not None, f"Subject {s} not found registered."

    def load_single_session(self, session_path: Path | str):
        if isinstance(session_path, str):
            session_path = Path(session_path)
        if not (hash_dir := self.preprocess_path(session_path)).exists():
            loader = get_loader(session_path)
            meta = loader(session_path, self.cfg, hash_dir)
            self.validate_meta(meta)
            meta.to_csv(hash_dir / 'meta.csv')
        else:
            meta = pd.read_csv(hash_dir / 'meta.csv')
        return meta

    def __getitem__(self, index):
        trial = self.meta_df.iloc[index]
        # * Potential optimization point to load onto GPU directly
        payload = torch.load(trial.path)
        return {
            **{k: payload[k] for k in self.cfg.data_keys},
            **{k: self.context_index[k].index(trial[k]) for k in self.cfg.meta_keys}
        }

    def build_context_index(self):
        logging.info("Building context index; any previous DataAttrs may be invalidated.")
        context = {}
        for k in self.cfg.meta_keys:
            assert k in self.meta_df.columns, f"Key {k} not in metadata"
            import pdb;pdb.set_trace()
            context[str(k)] = sorted(self.meta_df[k].unique()) # convert key from enum so we can build contextattrs
            # TODO not obvious cast is needed
            # TODO not obvious we can even index meta_df correctly
        self.context_index: Dict[str, List] = context

    def get_data_attrs(self):
        r"""
            Provide information about unique context such as
            - participants in dataset (and array information)
            - sessions used
            - tasks attempted.
            To be consumed by model to determine model IO.
        """
        if self.context_index is None:
            self.build_context_index()
        return DataAttrs(
            bin_size_ms=self.cfg.bin_size_ms,
            context=ContextAttrs(**self.context_index)
        )

    # ==================== Data splitters ====================
    @property
    def split_keys(self):
        return self.meta_df[self.cfg.split_key].unique().copy()

    def get_key_indices(self, key_values, key: MetaKey=MetaKey.unique):
        return self.meta_df[self.meta_df[key].isin(key_values)].index

    @property
    def subset_by_key(self, key_values: List[Any], key: MetaKey=MetaKey.unique, allow_second_subset=True):
        r"""
            # ! In place
            # (Minimum generalization is, same trial, different pulse)
            # To new trials, likely practical minimum
            # To new channel, amplitudes, etc. in the future
            Note - does not update `self.session` as we want to track the source (multi-)sessions to prevent unintended session mixup (see `merge`)
        """
        if len(key_values) == 0:
            logging.info("No keys provided, ignoring subset.")
            return
        if self.subsetted:
            assert allow_second_subset
            logging.warn("Dataset has already been subsetted.")
        self.meta_df = self.meta_df[self.meta_df[key].isin(key_values)]
        self.meta_df = self.meta_df.reset_index(drop=True)
        self.build_context_index()
        self.subsetted = True

    def tv_split_by_split_key(self, train_ratio=0.8, seed=None):
        keys = self.split_keys
        if seed is None:
            seed = self.cfg.dataset_seed
        pl.seed_everything(seed)
        np.random.shuffle(keys)
        tv_cut = int(train_ratio * len(keys))
        train_keys, val_keys = keys[:tv_cut], keys[tv_cut:]
        return train_keys, val_keys

    def create_tv_datasets(self, **kwargs):
        r"""
            Keys determine how we split up our dataset.
            Default by trial, or more specific conditions
            Assumes balanced dataset
        """
        train_keys, val_keys = self.tv_split_by_split_key(**kwargs)
        train = copy.deepcopy(self)
        train.subset_by_key(train_keys, key=self.cfg.split_key)
        val = copy.deepcopy(self)
        val.subset_by_key(val_keys, key=self.cfg.split_key)
        return train, val

    def merge(self, data_other: Self):
        self.meta_df = pd.concat([self.meta_df, data_other.meta_df])
        self.meta_df = self.meta_df.reset_index(drop=True)
        self.build_context_index()
        # TODO think about resetting data attrs - this should be called before any data attr call

