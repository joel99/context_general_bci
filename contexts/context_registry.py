# Information about the real world.
# Includes experimental notes, in lieu of readme
# Ideally, this class can be used outside of this specific codebase.

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import functools

from .context_info import ContextInfo, ReachingContextInfo, PassiveICMSContextInfo
from tasks import ExperimentalTask
r"""
    ContextInfo class is an interface for storing meta-information needed by several consumers, mainly the model, that may not be logged in data from various sources.
    ContextRegistry allows consumers to query for this information from various identifying sources.
    Note - external registry calls should use the instance, not the class.
    This appears to be necessary for typing to work more reliably (unclear).

    Binds subject, task, and other metadata like datapath together.
    JY's current view of  ideal dependency tree is
    Consumer (e.g. model training loop) -- depends -- > ContextRegistry --> Task -> Subject Registry
    - (But currently consumer will dive into some task-specific details, loader could be refactored..)
"""

r"""
    To support a new task
    - Add a new enum value to ExperimentalTask
    - Add experimental config to DatasetConfig
    - Subclass ContextInfo and implement the abstract methods
"""

class ContextRegistry:
    instance = None
    _registry: Dict[str, ContextInfo] = {}
    search_index = None  # allow multikey querying

    def __new__(cls, init_items: List[ContextInfo]=[]):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.search_index = pd.DataFrame()
            cls.instance.register(init_items)
        return cls.instance

    def build_search_index(self, items: List[ContextInfo]):
        index = [{
            'id': item.id,
            'task': item.task,
            'datapath': item.datapath,
            **item.get_search_index()
        } for item in items]
        return pd.DataFrame(index)

    # ! Note, current pattern is to put all experiments in a big list below; not use this register handle.
    def register(self, context_info: List[ContextInfo]):
        self.search_index = pd.concat([self.search_index, self.build_search_index(context_info)])
        for item in context_info:
            self._registry[item.id] = item

    def query(self, **search) -> ContextInfo | None:
        def search_query(df):
            return functools.reduce(lambda a, b: a & b, [df[k] == search[k] for k in search])
        queried = self.search_index.loc[search_query]
        if len(queried) == 0:
            return None
        elif len(queried) > 1:
            raise ValueError(f"Multiple contexts found for {search}")
        return self._registry[queried['id'].values[0]]

    def query_by_datapath(self, datapath: Path) -> ContextInfo:
        found = self.search_index[self.search_index.datapath == datapath]
        assert len(found) == 1
        return self._registry[found.iloc[0]['id']]

    def query_by_id(self, id: str) -> ContextInfo:
        return self._registry[id]

context_registry = ContextRegistry([
    PassiveICMSContextInfo.build(880, 1, 'stim_trains_gen2_chan34-40_80uA_0.5ITI_36cond/'),
    PassiveICMSContextInfo.build(906, 1, 'stim_trains_gen4-02b-ant_chan14-19-20-25_80uA_0.5ITI_6cond/'),

    PassiveICMSContextInfo.build(980, 4, 'stim_trains_additivity_chan34-36-45-47-49-50-51-52_80uA_0.5ITI_12cond'), # Not yet analyzed
    PassiveICMSContextInfo.build(985, 1, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_0'),

    PassiveICMSContextInfo.build(48, 1, 'stim_trains_80uA_9rap_9std/', (2, 6)),

    PassiveICMSContextInfo.build(61, 6, 'stim_trains_gen2_post_80uA_0.1ITI_36cond/'), # CRS07Lab
    PassiveICMSContextInfo.build(67, 1, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_0/'), # CRS07Lab
    PassiveICMSContextInfo.build(78, 1, 'stim_trains_gen6-07_chan14-19-20-25-10-15-18-12_80uA_0.5ITI_40cond'),
    PassiveICMSContextInfo.build(79, 3, 'stim_trains_psth-test_chan34-37-40-43_80uA_0.5ITI_2cond'),
    PassiveICMSContextInfo.build(82, 4, 'stim_trains_gen6-07_chan14-19-20-25-10-15-18-12_80uA_0.5ITI_40cond'),
    PassiveICMSContextInfo.build(88, 3, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_1/'),
    PassiveICMSContextInfo.build(91, 4, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_2/'),
    PassiveICMSContextInfo.build(92, 6, 'stim_trains_scaling-test_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_8cond/'),
    PassiveICMSContextInfo.build(98, 5, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_3/'),
    PassiveICMSContextInfo.build(105, 4, 'stim_trains_gen3-07_chan40-45-49-35-42-55-47-50-44_80uA_0.5ITI_8cond'),
    PassiveICMSContextInfo.build(107, 3, 'stim_trains_gen3-07_chan1-27-5-30-11-31-17-12-19_80uA_0.5ITI_8cond'),
    PassiveICMSContextInfo.build(120, 3, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_4/'),
    PassiveICMSContextInfo.build(120, 4, 'stim_trains_single-07-post_chan50-44-56-34_80uA_0.5ITI_4cond'),

    PassiveICMSContextInfo.build(126, 3, "", stim_banks=(6,)), # Not arbitrary stim, detection calibration
    PassiveICMSContextInfo.build(126, 5, "", stim_banks=(6,)), # Not arbitrary stim, detection decoding (~40 trials),
    PassiveICMSContextInfo.build(128, 3, 'stim_trains_gen4-07-post_chan46-51-52-57_80uA_0.5ITI_6cond'),
    PassiveICMSContextInfo.build(131, 3, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_0'),
    PassiveICMSContextInfo.build(131, 4, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_5'), # VISUAL DECODING
    PassiveICMSContextInfo.build(132, 3, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_1'), # VISUAL DECODING


    ReachingContextInfo.build('./data/nlb/000128/sub-Jenkins', ExperimentalTask.maze, alias='mc_maze'),
    ReachingContextInfo.build('./data/nlb/000138/sub-Jenkins', ExperimentalTask.maze, alias='mc_maze_large'),
    ReachingContextInfo.build('./data/nlb/000139/sub-Jenkins', ExperimentalTask.maze, alias='mc_maze_medium'),
    ReachingContextInfo.build('./data/nlb/000140/sub-Jenkins', ExperimentalTask.maze, alias='mc_maze_small'),
    ReachingContextInfo.build('./data/nlb/000129/sub-Indy', ExperimentalTask.rtt, alias='mc_rtt'),
])
