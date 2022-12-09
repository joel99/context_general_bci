import abc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np

import logging

from utils import StimCommand

from config import DatasetConfig
from subjects import SubjectArrayRegistry, SubjectInfo, SubjectName
from subjects.pitt_chicago import CRS02b, CRS07
from tasks import ExperimentalTask, ExperimentalTaskRegistry

# FYI: Inherited dataclasses don't call parent's __init__ by default. This is a known issue/feature:
# https://bugs.python.org/issue43835

@dataclass(kw_only=True)
class ContextInfo:
    r"""
        Base (abstract) class for static info for a given dataset.
        Subclasses should specify identity and datapath
    """
    # Context items - this info should be provided in all datasets.
    subject: SubjectInfo # note this is an object/value
    task: ExperimentalTask # while this is an enum/key, currently

    # These should be provided as denoted in SubjectArrayRegistry WITHOUT subject specific handles.
    # Dropping subject handles is intended to be a convenience since these contexts are essentially metadata management. TODO add some safety in case we decide to start adding handles explicitly as well...
    _arrays: List[str] = field(default_factory=lambda: []) # arrays (without subject handles) that were active in this context. Defaults to all known arrays for subject


    datapath: Path = Path("fake_path") # path to raws - to be provided by subclass (not defaulting to None for typing)
    alias: str = ""

    def __init__(self,
        subject: SubjectInfo,
        task: str,
        _arrays: List[str] = [],
        alias: str = "",
        **kwargs
    ):
        self.subject = subject
        self.task = task
        self.alias = alias
        # This is more or less an abstract method; not ever intended to be run directly.

        # self.build_task(**kwargs) # This call is meaningless since base class __init__ isn't called
        # Task-specific info are responsible for assigning self.datapath

    def __post_init__(self):
        if not self._arrays: # Default to all arrays
            self._arrays = self.subject.arrays.keys()
        else:
            assert all([self.subject.has_array(a) for a in self._arrays]), \
                f"An array in {self._arrays} not found in SubjectArrayRegistry"
        assert self.datapath is not Path("fake_path"), "ContextInfo didn't initialize with datapath"
        if not self.datapath.exists():
            logging.warn(f"ContextInfo datapath not found ({self.datapath})")

    @property
    def array(self) -> List[str]:
        r"""
            We wrap the regular array ID with the subject so we don't confuse arrays across subjects.
            These IDs will be used to query for array geometry later. `array_registry` should symmetrically register these IDs.
        """
        return [self.subject.wrap_array(a) for a in self._arrays]

    @property
    def id(self):
        return f"{self.task}-{self.subject.name}-{self._id()}"

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

    def load(self, cfg: DatasetConfig, cache_root: Path):
        loader = ExperimentalTaskRegistry.get_loader(self.task)
        return loader.load(
            self.datapath,
            cfg=cfg,
            cache_root=cache_root,
            subject=self.subject,
            context_arrays=self.array,
            dataset_alias=self.alias
        )

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
        info = PassiveICMSContextInfo(
            subject=CRS02b if session > 500 else CRS07,
            task=ExperimentalTask.passive_icms,
            _arrays=["medial_s1", "lateral_s1"],
            session=session,
            set=set,
            train_dir=train_dir,
            stim_banks=stim_banks,
            stimsync_banks=stimsync_banks,
            datapath=Path(f"/home/joelye/projects/icms_modeling/data/binned_pth/{session:05d}.Set{set:04d}.full.pth")
        )
        info.build_task(
            session=session,
            set=set,
            train_dir=train_dir,
            stim_banks=stim_banks,
            stimsync_banks=stimsync_banks,
            stim_train_dir_root=stim_train_dir_root,
        ) # override
        return info

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
        return f"{self.session}-{self.alias}" # All reaching data get alias

    @classmethod
    def build(cls, datapath_str: str, task: ExperimentalTask, alias: str="", arrays=["main"]):
        datapath = Path(datapath_str)
        subject = SubjectArrayRegistry.query_by_subject(
            datapath.name.split('-')[-1].lower()
        )
        session = int(datapath.parent.name)
        return ReachingContextInfo(
            subject=subject,
            task=task,
            _arrays=arrays,
            alias=alias,
            session=session,
            datapath=datapath,
        )

    @classmethod
    def build_several(cls, datapath_folder_str: str, task: ExperimentalTask, alias_prefix: str = "", arrays=["PMd", "M1"]):
        # designed around churchland reaching data
        datapath_folder = Path(datapath_folder_str)
        subject = SubjectArrayRegistry.query_by_subject(
            datapath_folder.name.split('-')[-1].lower()
        )
        session = int(datapath_folder.parent.name)
        all_info = []
        for i, path in enumerate(datapath_folder.glob("*.nwb")):
            alias = f"{alias_prefix}-{i}" if alias_prefix else f"reaching-{subject.name}-{i}"
            all_info.append(ReachingContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=alias,
                session=session,
                datapath=path,
            ))
        return all_info

    def get_search_index(self):
        return {
            **super().get_search_index(),
            'session': self.session,
        }

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