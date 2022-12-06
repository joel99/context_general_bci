# Information about the real world.
# Includes experimental notes, in lieu of readme
# Ideally, this class can be used outside of this specific codebase.

import abc
from dataclasses import dataclass, field
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import functools

from utils import StimCommand

from subjects import SubjectArrayRegistry, SubjectName
from tasks import ExperimentalTaskRegistry, ExperimentalTask
r"""
    ContextInfo class is an interface for storing meta-information needed by several consumers, mainly the model, that may not be logged in data from various sources.
    ContextRegistry allows consumers to query for this information from various identifying sources.
"""

r"""
    To support a new task
    - Add a new enum value to ExperimentalTask
    - Add experimental config to DatasetConfig
    - Subclass ContextInfo and implement the abstract methods
"""

@dataclass(kw_only=True)
class ContextInfo:
    r"""
        Base (abstract) class for static info for a given dataset.
        Subclasses should specify identity and datapath
    """
    # Context items - this info should be provided in all datasets.
    subject: str # These should match what's registered in SubjectArrayRegistry
    task: str

    # These should be provided as denoted in SubjectArrayRegistry WITHOUT subject specific handles.
    # Dropping subject handles is intended to be a convenience since these contexts are essentially metadata management. TODO add some safety in case we decide to start adding handles explicitly as well...
    _arrays: List[str] = field(default_factory=lambda: []) # arrays (without subject handles) that were active in this context. Defaults to all known arrays for subject


    datapath: Path = Path("fake_path") # path to raws - to be provided by subclass (not defaulting to None for typing)
    alias: str = ""

    def __init__(self,
        subject: str,
        task: str,
        _arrays: List[str] = [],
        alias: str = "",
        **kwargs
    ):
        assert SubjectArrayRegistry.query_by_subject(subject) is not None, f"Subject {subject} not found in SubjectArrayRegistry"
        self.subject = subject
        self.task = task
        self.alias = alias
        if not _arrays: # Default to all arrays
            self._arrays = SubjectArrayRegistry.query_by_subject(subject).arrays.values()
        else:
            assert all([
                SubjectArrayRegistry.query_by_array(SubjectArrayRegistry.wrap_array(self.subject, a)) is not None for a in _arrays
                ]), f"Arrays {_arrays} not found in SubjectArrayRegistry"
            self._arrays = _arrays

        self.build_task(**kwargs)
        # Task-specific builders are responsible for assigning self.datapath
        assert self.datapath is not None and self.datapath.exists(), "ContextInfo must be built with a valid datapath"

    @property
    def arrays(self):
        r"""
            We wrap the regular array ID with the subject so we don't confuse arrays across subjects.
            These IDs will be used to query for array geometry later. `array_registry` should symmetrically register these IDs.
        """
        return [SubjectArrayRegistry.wrap_array(self.subject, a) for a in self._arrays]

    @property
    def id(self):
        return f"{self.task}-{self.subject}-{self._id()}"

    @abc.abstractmethod
    def _id(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def build_task(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def build_task(cls, **kwargs):
        raise NotImplementedError

    def get_search_index(self):
        # Optional method for allowing searching the registry with these keys
        return {
            'alias': self.alias,
            'subject': self.subject
        }

    def get_loader(self):
        return ExperimentalTaskRegistry.get_loader(self.task)

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


@dataclass
class PassiveICMSContextInfo(ContextInfo):
    session: int
    set: int

    train_dir: Path
    stim_banks: Tuple[int]
    stimsync_banks: Tuple[int]

    _bank_to_array_name = {
        2: "lateral_s1",
        6: "medial_s1"
    }

    @classmethod
    def calculate_stim_banks(cls, train_dir: Path) -> Tuple[int]:
        all_channels = []
        for stim_train_txt in train_dir.iterdir():
            if 'PULSE_TRAIN_' not in str(stim_train_txt):
                continue
            with open(stim_train_txt, 'r') as f:
                _, command_line, *_ = f.readlines()
                command_channels = [int(c) for c in command_line.split()[::2]]
                all_channels.extend(command_channels)
        all_channels = np.unique(np.array(all_channels))
        stim_banks = []
        if ((all_channels >= 1) & (all_channels < 33)).any():
            stim_banks.append(2)
        if ((all_channels >= 33) & (all_channels < 65)).any():
            stim_banks.append(6)
        return tuple(stim_banks)

    @classmethod
    def get_stim_banks(cls, train_dir: Path) -> Tuple[int]:
        bank_path = train_dir / "banks.txt"
        if not bank_path.exists():
            stim_banks = cls.calculate_stim_banks()
            with open(bank_path, 'w') as f:
                f.write(" ".join([str(b) for b in stim_banks]))
            return stim_banks
        with open(bank_path, 'r') as f:
            stim_banks = tuple([int(b) for b in f.read().split()])
        return stim_banks

    @classmethod
    def build(cls,
        session,
        set,
        train_dir,
        stim_banks=None,
        stimsync_banks=None,
        stim_train_dir_root=Path("/home/joelye/projects/icms_modeling/data/stim_trains/"),
    ):
        if stim_banks is None:
            stim_banks = [cls._bank_to_array_name[sb] for sb in cls.get_stim_banks(stim_train_dir_root / train_dir)]
        if stimsync_banks is None:
            stimsync_banks = ["medial_s1", "lateral_s1"]
            # stimsync_banks = stim_banks
        return PassiveICMSContextInfo(
            subject=SubjectName.CRS02b if session > 500 else SubjectName.CRS07,
            task=ExperimentalTask.passive_icms,
            _arrays=["medial_s1", "lateral_s1"],
            session=session,
            set=set,
            train_dir=train_dir,
            stim_banks=stim_banks,
            stimsync_banks=stimsync_banks,
        )

    def _id(self):
        return f"{self.session}_{self.set}"

    def build_task(self,
        session: int = 0,
        set: int = 0,
        train_dir: str = "",
        stim_banks=None,
        stim_train_dir_root=Path("/home/joelye/projects/icms_modeling/data/stim_trains/"),
        stimsync_banks=None,
        # We can be lazy here, but try not to accept kwargs - tasks shouldn't be getting unused args
    ):
        self.session = session
        self.set = set
        if train_dir == "":
            self.train_dir = ""
        else:
            self.train_dir = stim_train_dir_root / train_dir
        if stim_banks is None:
            self.stim_banks = self.get_stim_banks()
        else:
            self.stim_banks = stim_banks
        self.stimsync_banks = stimsync_banks if stimsync_banks else self.stim_banks
        self.stimsync_banks = (2, 6) # ! Ok. While stimsync technically shouldn't be affecting off-stim (e.g. the cereplexes), something is amiss - validation becomes way oversynced once I register the proper banks. Not worth investigating right now.
        # ! Thus, I will leave it as this. I suspect it's due to estimator failure.
        self.datapath = Path(f"/home/joelye/projects/icms_modeling/data/binned_pth/{self.session:05d}.Set{self.set:04d}.full.pth")

    def get_search_index(self):
        return {
            **super().get_search_index(),
            'session': self.session,
            'set': self.set,
        }

@dataclass
class ReachingContextInfo(ContextInfo):
    # NLB barebones
    session: int

    def _id(self):
        return f"{self.session}"

    @classmethod
    def build(cls, datapath_str: str, task: ExperimentalTask, alias: str=""):
        datapath = Path(datapath_str)
        subject = datapath.name.split('-')[-1].lower()
        session = int(datapath.parent.name)
        return ReachingContextInfo(
            subject=subject,
            task=task,
            alias=alias,
            session=session,
            datapath=datapath,
        )

    def build_task(self, session: int, datapath: Path):
        self.session = session
        self.datapath = datapath

    def get_search_index(self):
        return {
            **super().get_search_index(),
            'session': self.session,
        }


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

# ====
# Archive

# Preprocessing
session_config_kwargs = {
    61: {
        'infer_stimsync': True,
    },
    872: {
        'infer_stimsync': True,
    },
    878: {
        'infer_stimsync': True,
        'ref_channel': 33 # Actually assumed for both NSPs, not great.
    },
    884: {
        'infer_stimsync': True,
        'ignore_nev': True, # The one sample in this file appears irrelevant.
        # 'stimsync_padding': 4, # wider since we are inferring nearly whole trial
        'early_blank': True,
        'infer_stimsync_fallback_offset': 6, # Not 10, since early blank kills some padding
    },
    88: {
        'trial_offset': 1585
    },
    91: {
        'trial_offset': 3170
    },
    98: {
        'trial_offset': 4755
    },
    (120, 3): {
        'trial_offset': 6340
    },
    (131, 4): {
        'trial_offset': 7925
    },
    (131, 2): {
        'trial_offset': 1585
    }
}

# Trials with visually obvious defects or some uncontrolled externality of note
session_trial_blacklist = defaultdict(list) # (Trial nums, not index) # ! Note must be raw (not offset) recorded trial
session_trial_blacklist[886] = [1]


def extract_commanded_stim(stim_dir, trial) -> StimCommand:
    stim_txt = Path(stim_dir) / f"PULSE_TRAIN_trial{trial:04d}.txt"
    if not stim_txt.exists():
        stim_txt = Path(stim_dir) / f"PULSE_TRAIN_RAP_trial{trial:04d}.txt" # legacy
    with open(str(stim_txt), 'r') as f:
        times = f.readline().split()[::2]
        channel_current = f.readline().split()
        times = np.array(times, dtype=float)
        channels = np.array(channel_current[::2], dtype=int)
        currents = np.array(channel_current[1::2], dtype=int)
    return StimCommand(times[channels != 0], channels[channels != 0], currents[channels != 0])