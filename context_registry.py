# Information about the real world.
# Includes experimental notes, in lieu of readme
from dataclasses import dataclass
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from config import ExperimentalTask
from utils import StimCommand

from array_registry import subject_array_registry, SubjectArrayRegistry
r"""
    ContextInfo class is an interface for storing meta-information needed by several consumers, mainly the model, that may not be logged in data from various sources.
    ContextRegistry allows consumers to query for this information from various identifying sources.
"""

@dataclass
class ContextInfo:
    r"""
        Holds static info for a given dataset.

        TODO currently built around ICMS, refactor/subclass to be more general
    """
    # Context items - this info should be provided in all datasets.
    session: int
    subject: str # These should match what's registered in SubjectArrayRegistry
    task: str
    # These should be provided as denoted in SubjectArrayRegistry WITHOUT subject specific handles.
    # Dropping subject handles is intended to be a convenience since these contexts are essentially metadata management. TODO add some safety in case we decide to start adding handles explicitly as well...
    _arrays: List[str]

    stim_banks: Tuple[int]
    stimsync_banks: Tuple[int]
    set: int = 0
    train_dir: Path = ""

    def __init__(self,
        session: str, # TODO this isn't the right abstraction - we want some sort of ID
        subject: str,
        task: str,
        arrays: List[str] = [],
        **kwargs
    ):
        self.session = session
        assert subject_array_registry.query_by_subject(subject) is not None, f"Subject {subject} not found in SubjectArrayRegistry"
        self.subject = subject
        self.task = task # TODO build task registry
        if not arrays: # Default to all arrays
            self._arrays = subject_array_registry.query_by_subject(subject).arrays.values()
        else:
            self._arrays = arrays


        # TODO - pitt-type experiment
        self.set = 0

        if self.task == ExperimentalTask.passive_icms:
            self.build_icms(**kwargs)
        # Task-specific builders are responsible for assigning this
        assert self.datapath is not None and self.subject is not None, "ContextInfo must be built with a valid datapath and subject"
        assert self.datapath.exists(), f"datapath {self.datapath} does not exist"

    @property
    def id(self):
        if self.task == ExperimentalTask.passive_icms:
            return f"{self.subject}_{self.session}_{self.set}"

    @property
    def arrays(self):
        r"""
            We wrap the regular array ID with the subject so we don't confuse arrays across subjects.
            These IDs will be used to query for array geometry later. `array_registry` should symmetrically register these IDs.
        """
        return [SubjectArrayRegistry.wrap_array(self.subject, a) for a in self._arrays]

    # ICMS props
    def build_icms(self,
        train_dir: str = "",
        stim_banks=None,
        stim_train_dir_root=Path("/home/joelye/projects/icms_modeling/data/stim_trains/"),
        stimsync_banks=None,
        **kwargs
    ):
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
        self.subject = "CRS02b" if self.session > 500 else "CRS07" # Infer

    def get_stim_banks(self) -> Tuple[int]:
        bank_path = self.train_dir / "banks.txt"
        if not bank_path.exists():
            stim_banks = self.calculate_stim_banks()
            with open(bank_path, 'w') as f:
                f.write(" ".join([str(b) for b in stim_banks]))
            return stim_banks
        with open(bank_path, 'r') as f:
            stim_banks = tuple([int(b) for b in f.read().split()])
        return stim_banks

    def calculate_stim_banks(self) -> Tuple[int]:
        all_channels = []
        for stim_train_txt in self.train_dir.iterdir():
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

class ContextRegistry:
    instance = None
    _registry: Dict[str, ContextInfo] = {}
    search_index = None  # allow multikey querying

    def __new__(cls, init_items: List[ContextInfo]=[]):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
            cls.search_index = pd.DataFrame()
            cls.instance.register(init_items)
        return cls.instance

    def build_search_index(self, items: List[ContextInfo]):
        index = [(item.id, item.session, item.set, item.datapath) for item in items]
        return pd.DataFrame(index, columns=['id', 'session', 'set', 'datapath'])

    def register(self, context_info: List[ContextInfo]):
        self.instance.search_index = pd.concat([self.instance.search_index, self.instance.build_search_index(context_info)])
        for item in context_info:
            self.instance._registry[item.id] = item

    # For Pittsburgh human trials, this uniquely identifies a session
    def query_by_session_set(self, session: int, set: Optional[int] = None) -> ContextInfo:
        if set is None:
            found = self.instance.search_index[self._registry.session == session]
            if len(found) > 1:
                raise ValueError(f"Multiple sets found for session {session}, please specify set")
        else:
            found = self.instance.search_index[(self.search_index.session == session) & (self.instance.search_index.set == set)]
            assert len(found) == 1
        return self._registry(found.iloc[0]['id'])

    def query_by_datapath(self, datapath: Path) -> ContextInfo:
        found = self.instance.search_index[self.search_index.datapath == datapath]
        assert len(found) == 1
        return self._registry(found.iloc[0]['id'])

    def query_by_id(self, id: str) -> ContextInfo:
        return self._registry[id]


context_registry = ContextRegistry([
    # TODO update all of these... probably via an ICMS subclass that binds sensory arrays
    ContextInfo(845, 9, 'stim_trains_100uA_500ITI_long/', (2, 6)),
    ContextInfo(850, 7, 'stim_trains_single_pulse/', (2, 6)),
    ContextInfo(853, 4, 'stim_trains_100uA_500ITI_16cond_padded/', (2, 6)),
    ContextInfo(872, 21, 'stim_trains_80uA_9rap_9std/', (2, 6)),
    ContextInfo(874, 4, 'stim_trains_rap_std_chan19-29-34-40_80uA_0.5ITI_18cond/', stimsync_banks=(2, 6)),
    ContextInfo(878, 4, 'stim_trains_gen2_chan19-29_80uA_0.5ITI_36cond/', stimsync_banks=(2, 6)),
    ContextInfo(880, 1, 'stim_trains_gen2_chan34-40_80uA_0.5ITI_36cond/'),
    ContextInfo(884, 11, 'stim_trains_gen2_chan19-29_80uA_0.5ITI_36cond/', stimsync_banks=(2, 3)),
    ContextInfo(886, 2, 'stim_trains_gen3-02b-ant_chan18-12-24-2-20-29-14-25-19_80uA_0.5ITI_8cond/', stimsync_banks=(2, 3)),
    ContextInfo(906, 1, 'stim_trains_gen4-02b-ant_chan14-19-20-25_80uA_0.5ITI_6cond/'),

    ContextInfo(980, 4, 'stim_trains_additivity_chan34-36-45-47-49-50-51-52_80uA_0.5ITI_12cond'), # Not yet analyzed
    ContextInfo(985, 1, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_0'),

    ContextInfo(48, 1, 'stim_trains_80uA_9rap_9std/', (2, 6)),

    # Session 62 uses V1 trains (which doesn't distinguish different channels as different conditions. Note no channel annotation) Cond 1-9 are std, 10-27 are RAP with repeats, 28-36 are RAP without repeats. Train with 19-27, 28-36, target 1-9, 10-17.
    ContextInfo(62, 5, 'stim_trains_gen2_post_80uA_0.1ITI_36cond/', stimsync_banks=(2, 6)),
    ContextInfo(61, 6, 'stim_trains_gen2_post_80uA_0.1ITI_36cond/'), # CRS07Lab
    ContextInfo(67, 1, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_0/'), # CRS07Lab
    ContextInfo(78, 1, 'stim_trains_gen6-07_chan14-19-20-25-10-15-18-12_80uA_0.5ITI_40cond'),
    ContextInfo(79, 3, 'stim_trains_psth-test_chan34-37-40-43_80uA_0.5ITI_2cond'),
    ContextInfo(82, 4, 'stim_trains_gen6-07_chan14-19-20-25-10-15-18-12_80uA_0.5ITI_40cond'),
    ContextInfo(88, 3, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_1/'),
    ContextInfo(91, 4, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_2/'),
    ContextInfo(92, 6, 'stim_trains_scaling-test_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_8cond/'),
    ContextInfo(98, 5, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_3/'),
    ContextInfo(105, 4, 'stim_trains_gen3-07_chan40-45-49-35-42-55-47-50-44_80uA_0.5ITI_8cond'),
    ContextInfo(107, 3, 'stim_trains_gen3-07_chan1-27-5-30-11-31-17-12-19_80uA_0.5ITI_8cond'),
    ContextInfo(120, 3, 'stim_trains_scaling-train_chan34-36-42-44-46-47-50-51-52-55-56-57_80uA_0.5ITI_1cond/block_4/'),
    ContextInfo(120, 4, 'stim_trains_single-07-post_chan50-44-56-34_80uA_0.5ITI_4cond'),

    ContextInfo(126, 3, "", stim_banks=(6,)), # Not arbitrary stim, detection calibration
    ContextInfo(126, 5, "", stim_banks=(6,)), # Not arbitrary stim, detection decoding (~40 trials),
    ContextInfo(128, 3, 'stim_trains_gen4-07-post_chan46-51-52-57_80uA_0.5ITI_6cond'),
    ContextInfo(131, 3, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_0'),
    ContextInfo(131, 4, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_5'), # VISUAL DECODING
    ContextInfo(132, 3, 'stim_trains_scaling-train_chan2-4-10-12-14-15-18-19-20-23-24-25_80uA_0.5ITI_1cond/block_1'), # VISUAL DECODING
])


# ====

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

def infer_s874_command(
    stim_samples: List[int],
    voltage_db_channel: int,
    voltage_db_amp: int,
    guess_limit=100,
) -> Tuple[StimCommand, int]:
    r"""
        Session 874 was commanded with a shuffled version of the trials.
        Thus we need to infer what exactly was used on any given trial using additional data pulled from `infer_voltage_db`
        Returns command, and condition label
    """
    STD_AMPS = [20, 50, 80]
    STD_PULSES = [21, 51, 81]
    CHANNELS = [19, 29, 34, 40]
    UNIQUE_RAP_SAMPLES = [38, 39, 42, 44, 47, 48, 56, 57, 64]
    stim_dir = session_info[874].train_dir
    for trial_guess in range(1, guess_limit):
        command = extract_commanded_stim(stim_dir, trial_guess)
        try:
            condition_channels = CHANNELS.index(voltage_db_channel)
            if voltage_db_amp != 0: # std, we just need to find the matching standard to get the number
                if (
                    command.current[0] == command.current[1] == command.current[2] == command.current[3] == voltage_db_amp
                ) and len(stim_samples) == len(command.times):
                    condition_amp = STD_AMPS.index(voltage_db_amp)
                    condition_freq = STD_PULSES.index(len(stim_samples))
                    condition = condition_channels * 9 + condition_freq * 3 + condition_amp
                    return StimCommand(
                        command.times,
                        np.full_like(command.channels, voltage_db_channel),
                        command.current
                    ), condition
            else: # rap - in s874, # of pulses is unique in 9 rap conditions, so just check length
                # Offset by the number of standard conditions (36)
                if len(command.times) == len(stim_samples):
                    return StimCommand(
                        command.times,
                        np.full_like(command.channels, voltage_db_channel),
                        command.current
                    ), 36 + condition_channels * 9 + UNIQUE_RAP_SAMPLES.index(len(stim_samples))
        except Exception as e:
            raise Exception(f's874 inference failed: {e}')
    raise Exception('no match found')
