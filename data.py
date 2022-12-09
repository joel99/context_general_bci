from typing import List, Any, Optional, Dict
import copy
import json
import os
from pathlib import Path
import re
import itertools
from datetime import datetime
from dataclasses import dataclass, field

import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from config import DatasetConfig, MetaKey, DataKey
from subjects import SubjectArrayRegistry
from contexts import context_registry, ContextInfo
from tasks import ExperimentalTaskRegistry

r"""
    Stores range of contexts provided by a dataset.
    Data will serve attributes as index of provided contexts.
    The model should probably unique parameters for each context (JY thinks they should be embeddings).
    - `subject` context will _determine_ data shape.
    1. Simple is to get a linear layer per subject.
    2. Odd but maybe workable is to pad to some max length (e.g. 128, current generation Utah array).
    3. Stretch; too much of a separate research endeavor -- use Set networks to learn a subject-based cross-attention operator to downproject.
    These contexts are not independent. In fact, they're almost hierarchical.
    Subject -> Array -> Session, Task -> session.
"""

# Padding tokens
LENGTH_KEY = 'length'
CHANNEL_KEY = 'channel_counts'
@dataclass
class ContextAttrs:
    subject: List[str] = field(default_factory=list)
    array: List[str] = field(default_factory=list) # should be prefixed with subject
    session: List[str] = field(default_factory=list) # unique ID
    task: List[str] = field(default_factory=list) # experimental task

@dataclass
class DataAttrs:
    bin_size_ms: int
    spike_dim: int
    max_channel_count: int
    context: ContextAttrs
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
            meta_df.extend(self.load_session(d))
        self.meta_df = pd.concat(meta_df).reset_index()

        self.context_index = None
        self.subsetted = False

    @staticmethod
    def preprocess_path(cfg: DatasetConfig, session_path: Path) -> Path:
        return cfg.root_dir / cfg.preprocess_suffix / session_path.relative_to(cfg.root_dir)

    def validate_meta(self, meta_df: pd.DataFrame):
        for k in self.cfg.meta_keys:
            if k == MetaKey.subject:
                unique_subjects = meta_df[MetaKey.subject.name].unique()
                for s in unique_subjects:
                    assert SubjectArrayRegistry.query_by_subject(s) is not None, f"Subject {s} not found registered."
            elif k == MetaKey.array:
                pass # no validation
            else:
                assert k in meta_df.columns, f"Requested meta key {k} not loaded in meta_df"

    def preproc_version(self):
        return {
            'max_trial_length': self.cfg.max_trial_length,
            'bin_size_ms': self.cfg.bin_size_ms
        }

    def checksum_diff(self, version_path: Path):
        # load json in session path
        if not version_path.exists():
            return True
        with open(version_path, 'r') as f:
            cached_preproc_version = json.load(f)
        return self.preproc_version() != cached_preproc_version


    def load_session(self, session_path_or_alias: Path | str) -> List[pd.DataFrame]:
        r"""
            Data will be pre-cached, typically in a flexible format, so we don't need to regenerate too often.
            That is, the data and metadata will not adopt specific task configuration at time of cache, but rather try to store down all info available.
            The exception to this is the crop length.
        """

        if isinstance(session_path_or_alias, str):
            # Try alias
            context_meta = context_registry.query(alias=session_path_or_alias)
            if context_meta is None:
                session_path = Path(session_path_or_alias)
                context_meta = context_registry.query_by_datapath(session_path)
            else:
                if isinstance(context_meta, list):
                    logging.info('Multiple contexts found for alias, re-routing to several load calls.')
                    return [self.load_single_session(c) for c in context_meta]
        else:
            session_path = session_path_or_alias
            context_meta = context_registry.query_by_datapath(session_path)
        return [self.load_single_session(context_meta)]

    def load_single_session(self, context_meta: ContextInfo):
        session_path = context_meta.datapath
        if not (hash_dir := self.preprocess_path(self.cfg, session_path)).exists() or \
            self.checksum_diff(hash_dir / 'preprocess_version.json'):
            # TODO consider filtering meta df to be more lightweight (we don't bother right now because some nonessential attrs can be useful for analysis)
            os.makedirs(hash_dir, exist_ok=True)
            meta = context_meta.load(self.cfg, hash_dir)
            meta.to_csv(hash_dir / 'meta.csv')
            with open(hash_dir / 'preprocess_version.json', 'w') as f:
                json.dump(self.preproc_version(), f)
        else:
            meta = pd.read_csv(hash_dir / 'meta.csv')
            del meta[f'Unnamed: 0'] # index column
        for k in self.cfg.meta_keys:
            if k == MetaKey.array:
                context_array = getattr(context_meta, k.name)
                # Filter arrays using task configuration
                task_arrays = getattr(self.cfg, context_meta.task.name).arrays
                if task_arrays:
                    context_array = [a for a in context_array if a in task_arrays]
                for i in range(self.cfg.max_arrays):
                    meta[f'array_{i}'] = context_array[i] if i < len(context_array) else ""
                if len(context_array) > self.cfg.max_arrays:
                    logging.error(
                        f"Session {session_path} has more than {self.cfg.max_arrays} arrays."
                        f"Is this the right session? Or is max array setting to low?"
                        f"Or did you remember to truncate used arrays in task configuration?"
                    )
                    raise Exception()
            elif k == MetaKey.session:
                # never conflate sessions
                meta[k] = context_meta.id
            elif k == MetaKey.unique:
                continue # filled below
        meta[MetaKey.unique] = meta[MetaKey.session] + '-' + meta.index.astype(str)
        self.validate_meta(meta)

        return meta

    def __getitem__(self, index):
        r"""
            dict of arrays

            spikes: torch.Tensor, Batch x Time x Array x Channel x H
            * we give array dim (as opposed to flattening into channel to make array embeddings possible
        """
        trial = self.meta_df.iloc[index]
        # * Potential optimization point to load onto GPU directly
        meta_items = {}
        for k in self.cfg.meta_keys:
            if k == MetaKey.unique:
                continue # don't serve
            if k == MetaKey.array: # doing string comparisons probably isn't the fastest thing in the world
                meta_items[k] = torch.tensor([
                    self.context_index[k.name].index(trial[f'array_{i}']) for i in range(self.cfg.max_arrays)
                ])
            else:
                meta_items[k] = torch.tensor(self.context_index[k.name].index(trial[k])) # Casting in collater might be faster?

        r"""
            Currently we store spikes in a split-array format as a dict of tensors T C H.
            We must use the IDs to reconstruct the stack we want.
        """
        data_items = {}
        payload = torch.load(trial.path)
        channel_counts = []

        for k in self.cfg.data_keys:
            if k == DataKey.spikes:
                array_spikes = []
                # for alias in trial[MetaKey.array.name]:
                for i in range(self.cfg.max_arrays):
                    alias = trial[f'array_{i}']
                    alias_arrays = SubjectArrayRegistry.resolve_alias(alias) # list of strs
                    # ! Right now pad channels seems subservient to pad arrays, that doesn't seem to be necessary.
                    array_group = torch.cat([payload[k][a] for a in alias_arrays], dim=-2) # T C' H
                    if self.cfg.max_channels:
                        channel_counts.append(array_group.shape[-2])
                        array_group = torch.cat([
                            array_group, torch.zeros((
                                array_group.shape[0], self.cfg.max_channels - array_group.shape[-2], array_group.shape[2]
                            ), dtype=array_group.dtype)
                        ], -2) # T C H
                    array_spikes.append(array_group.unsqueeze(1))
                data_items[k] = torch.cat([
                    *array_spikes,
                    torch.zeros(
                        array_spikes[0].shape[0],
                        self.cfg.max_arrays - len(array_spikes),
                        *array_spikes[0].shape[2:],
                        dtype=array_spikes[0].dtype
                    )
                ], 1) # T A C H
                channel_counts.extend([0] * (self.cfg.max_arrays - len(array_spikes)))
            else:
                data_items[k] = payload[k]
        out = {
            **data_items,
            **meta_items,
        }
        if self.cfg.max_channels:
            out[CHANNEL_KEY] = torch.tensor(channel_counts) # of length arrays (subsumes array mask, hopefully)
        return out

    def __len__(self):
        return len(self.meta_df)

    def collater_factory(self):
        if not self.cfg.pad_batches:
            raise NotImplementedError("Need to implement trimming")
        def collater(batch):
            r"""
                batch: list of dicts
            """
            stack_batch = {}
            for k in batch[0].keys():
                if k == DataKey.spikes: # T A C H
                    stack_batch[LENGTH_KEY] = torch.tensor([b[k].shape[0] for b in batch])
                    stack_batch[k] = pad_sequence([b[k] for b in batch], batch_first=True)
                else:
                    stack_batch[k] = torch.stack([b[k] for b in batch], 0)
            return stack_batch
        return collater

    def build_context_index(self):
        logging.info("Building context index; any previous DataAttrs may be invalidated.")
        context = {}
        for k in self.cfg.meta_keys:
            if k == MetaKey.unique:
                continue # Only used as identifier, never served
            elif k == MetaKey.array:
                all_arrays = sorted(
                    pd.concat(self.meta_df[f'array_{i}'] for i in range(self.cfg.max_arrays)).unique()
                ) # This automatically includes the padding "" if it's present in df
                context[MetaKey.array.name] = all_arrays
            else:
                assert k in self.meta_df.columns, f"Key {k} not in metadata"
                context[k.name] = sorted(self.meta_df[k].unique()) # convert key from enum so we can build contextattrs
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
            max_channel_count=self.cfg.max_channels,
            spike_dim=1, # Higher dims not supported right now
            context=ContextAttrs(**self.context_index)
        )

    # ==================== Data splitters ====================
    @property
    def split_keys(self):
        return self.meta_df[self.cfg.split_key].unique().copy()

    def get_key_indices(self, key_values, key: MetaKey=MetaKey.unique):
        return self.meta_df[self.meta_df[key].isin(key_values)].index

    def subset_by_key(self, key_values: List[Any], key: MetaKey=MetaKey.unique, allow_second_subset=True):
        r"""
            # ! In place
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
        assert train.context_index == val.context_index, "Context index mismatch between train and val (some condition is unavailable, not supported)"
        return train, val

    def merge(self, data_other: Any): # should be type Self but surprisingly this is a 3.11 feature (I thought I used it before?)
        self.meta_df = pd.concat([self.meta_df, data_other.meta_df])
        self.meta_df = self.meta_df.reset_index(drop=True)
        self.build_context_index()
        # TODO think about resetting data attrs - this should be called before any data attr call
