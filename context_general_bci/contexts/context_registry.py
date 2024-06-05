# Information about the real world.
# Includes experimental notes, in lieu of readme
# Ideally, this class can be used outside of this specific codebase.

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import functools

from .context_info import ContextInfo

from context_general_bci.tasks import ExperimentalTask

CLOSED_LOOP_DIR = 'closed_loop_tests'
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
            'datapath': item.datapath.resolve(),
            **item.get_search_index()
        } for item in items]
        return pd.DataFrame(index)

    # ! Note, current pattern is to put all experiments in a big list below; not use this register handle.
    def register(self, context_info: List[ContextInfo]):
        context_info = [item for item in context_info if item is not None]
        self.search_index = pd.concat([self.search_index, self.build_search_index(context_info)])
        for item in context_info:
            self._registry[item.id] = item

    def query(self, **search) -> Union[ContextInfo, List[ContextInfo], None]:
        def search_query(df):
            non_str_search = [k for k in search if k != 'alias']
            if non_str_search:
                result = functools.reduce(lambda a, b: a & b, [df[k] == search[k] for k in non_str_search])
            else:
                result = pd.Series(True, index=df.index)
            if 'alias' not in search:
                return result
            if 'alias' not in df:
                print("no alias, nothing registered?")
                return None # test time
            return result & df['alias'].str.contains(search['alias'])

        queried = self.search_index.loc[search_query]
        if len(queried) == 0:
            return None
        elif len(queried) > 1:
            out = [self._registry[id] for id in queried['id']]
            return sorted(out)
        return self._registry[queried['id'].values[0]]

    def query_by_datapath(self, datapath: Union[Path, str]) -> ContextInfo:
        if not isinstance(datapath, Path):
            datapath = Path(datapath)
        found = self.search_index[self.search_index.datapath == datapath.resolve()]
        assert len(found) == 1, f"Found {len(found)} items for datapath {datapath}"
        return self._registry[found.iloc[0]['id']]

    def query_by_id(self, id: str) -> ContextInfo:
        return self._registry[id]

context_registry = ContextRegistry()

if not os.getenv('NDT_SUPPRESS_DEFAULT_REGISTRY', False):
    from .context_info import (
        ReachingContextInfo,
        PassiveICMSContextInfo,
        RTTContextInfo,
        DyerCOContextInfo,
        GallegoCOContextInfo,
        GDrivePathContextInfo,
        BCIContextInfo,
        BatistaContextInfo,
        FalconContextInfo,
    )
    context_registry.register([
        ReachingContextInfo.build('./data/nlb/000128/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze'),
        ReachingContextInfo.build('./data/nlb/000138/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_large'),
        ReachingContextInfo.build('./data/nlb/000139/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_medium'),
        ReachingContextInfo.build('./data/nlb/000140/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_small'),
        ReachingContextInfo.build('./data/nlb/000129/sub-Indy', ExperimentalTask.nlb_rtt, alias='mc_rtt'),

        *ReachingContextInfo.build_several('./data/churchland_reaching/000070/sub-Jenkins', ExperimentalTask.churchland_maze, alias_prefix='churchland_maze_jenkins'),
        *ReachingContextInfo.build_several('./data/churchland_reaching/000070/sub-Nitschke', ExperimentalTask.churchland_maze, alias_prefix='churchland_maze_nitschke'),

        # *ReachingContextInfo.build_several('./data/even_chen_delay/000121/sub-JenkinsC', ExperimentalTask.delay_reach, alias_prefix='even_chen_delay_jenkins'),
        # *ReachingContextInfo.build_several('./data/even_chen_delay/000121/sub-Reggie', ExperimentalTask.delay_reach, alias_prefix='even_chen_delay_reggie'),

        *RTTContextInfo.build_several('./data/odoherty_rtt/', alias_prefix='odoherty_rtt'),

        DyerCOContextInfo.build(('mihi', 1), ExperimentalTask.dyer_co, alias='dyer_co_mihi_1'),
        DyerCOContextInfo.build(('mihi', 2), ExperimentalTask.dyer_co, alias='dyer_co_mihi_2'),
        DyerCOContextInfo.build(('chewie', 1), ExperimentalTask.dyer_co, alias='dyer_co_chewie_1'),
        DyerCOContextInfo.build(('chewie', 2), ExperimentalTask.dyer_co, alias='dyer_co_chewie_2'),

        *GallegoCOContextInfo.build_from_dir('./data/gallego_co', task=ExperimentalTask.gallego_co),
        *GDrivePathContextInfo.build_from_dir('./data/churchland_misc'),

        *FalconContextInfo.build_from_dir('./data/h1/held_in_calib', task=ExperimentalTask.falcon_h1, suffix='calib', is_dandi=False),
        *FalconContextInfo.build_from_dir('./data/h1/held_out_calib', task=ExperimentalTask.falcon_h1, suffix='calib', is_dandi=False),

        # *FalconContextInfo.build_from_dir('./data/falcon/h1_oracle/', task=ExperimentalTask.falcon_h1, is_dandi=False),

        *FalconContextInfo.build_from_dir('./data/falcon/000941/sub-MonkeyL-held-in-calib', task=ExperimentalTask.falcon_m1),
        *FalconContextInfo.build_from_dir('./data/falcon/000941/sub-MonkeyL-held-out-calib', task=ExperimentalTask.falcon_m1),
        
        # *FalconContextInfo.build_from_dir('./data/falcon/m1/NWB_FALCON_v7_unsorted/held_out_oracle', task=ExperimentalTask.falcon_m1),
        
        *FalconContextInfo.build_from_dir('./data/falcon/000950/sub-T5-held-in-calib', task=ExperimentalTask.falcon_h2),
        *FalconContextInfo.build_from_dir('./data/falcon/000950/sub-T5-held-out-calib', task=ExperimentalTask.falcon_h2),
        
        *FalconContextInfo.build_from_dir('./data/falcon/000953/sub-MonkeyN-held-in-calib', task=ExperimentalTask.falcon_m2),
        *FalconContextInfo.build_from_dir('./data/falcon/000953/sub-MonkeyN-held-out-calib', task=ExperimentalTask.falcon_m2),
        
        # *FalconContextInfo.build_from_dir('./data/falcon/m2_oracle/', task=ExperimentalTask.falcon_m2),
        
        # *FalconContextInfo.build_from_dir('./data/m2/minival/', task=ExperimentalTask.falcon_m2, alias_prefix='minival_'),
        # *FalconContextInfo.build_from_dir('./data/falcon/000953/sub-MonkeyN-held-in-minival', task=ExperimentalTask.falcon_m2, alias_prefix='minival_'),

        # Not great to have this around, but debugging somethign critical
        # *FalconContextInfo.build_from_dir('./data/falcon/m1/eval', task=ExperimentalTask.falcon_m1),

        *RTTContextInfo.build_preproc('./data/calib/odoherty_rtt/', alias_prefix='calib_odoherty_calib_rtt'),
        *BCIContextInfo.build_preproc('./data/calib/pitt_co_alter', alias_prefix='calib_pitt_calib_broad_'), 
        *BCIContextInfo.build_preproc('./data/calib/pitt_grasp_alter', alias_prefix='calib_pitt_grasp_'), 

        *RTTContextInfo.build_preproc('./data/eval/odoherty_rtt/', alias_prefix='eval_odoherty_eval_rtt'),
        *BCIContextInfo.build_preproc('./data/eval/pitt_co_alter/', alias_prefix='eval_pitt_eval_broad_'),
        *BCIContextInfo.build_preproc('./data/eval/pitt_grasp_alter', alias_prefix='eval_pitt_grasp_'), 
        
    ])
