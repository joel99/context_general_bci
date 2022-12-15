import math
import re
from enum import Enum
from typing import Any, Iterable, List, Dict, Optional, Tuple, Union
from collections import defaultdict
from typing import NamedTuple
from typing_extensions import Self
from tqdm import tqdm
from multiprocessing import Pool
import os.path as osp
from pathlib import Path
import attr
import copy
import numpy as np
import pandas as pd
from yacs.config import CfgNode as CN
import pickle
from sklearn.model_selection import train_test_split

import scipy.signal as signal

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from subjects import subject_to_array

"""
TODO use CRS02 blacklist
"""
TrialNum = int
MetadataKey = str
ParticipantKey = str
Condition = int
ConditionInfo = Dict[Condition, List[int]] # Dict of condition to trials (zero-indexed)
ConditionDict = Dict[Condition, torch.Tensor]
SessionSet = Tuple[int, int]
class ICMSBatch(NamedTuple):
    record: torch.Tensor # B T C (D)
    stim: torch.Tensor # B T C (D)
    session: torch.Tensor # B
    teacher: Optional[torch.Tensor]
    lengths: Optional[torch.Tensor]

MAX_PREFIX_BUFFER = 1.4 # 1.4s
# MAX_PREFIX_BUFFER = 2e-1 # 200ms - affects where trial timeline begins relative to first pulse, affects cached data
MAX_DURATION = 2.4 # Max modeling length, including prefix. Note this form of cropping pays no heed to actual end of stim, but is fairly generous wrt ~300ms reaction time + <1s typical trials. Need to revisit if trials are very long (we'll lose a lot of data...)
# MAX_DURATION = 1.4 # Max modeling length, including prefix. Note this form of cropping pays no heed to actual end of stim, but is fairly generous wrt ~300ms reaction time + <1s typical trials. Need to revisit if trials are very long (we'll lose a lot of data...)
# ? One might think we could use StimData.duration, but I've got no idea what that points to... doesn't seem related to breadth of recorded spikes...
STIM_FLAG = 1
TIME_EPSILON = 1e-8 # in seconds
# Binning is numerically instable beyond 1e-10, so we align bins such that 1st stim begins bin_left[0] + epsilon. Be wary for binning effects on reproducibility.
SPIKE_TIMING_OFFSET = 1e-4 # 0.1ms
DATASET_SHUFFLE_KEY = 42
DEFAULT_SESSION_IDX = 0
# SPIKE_TIMING_OFFSET = 1.1e-3 # 1.1ms
# Stim times should be fixed at a reliable time in the bin, but can be unstable by up to 1.03ms (this max was found from 2 automatic surveys)
# The goal of this offset is to contain the entirely of the artifact (1.5ms) within one bin, but it's not really doable.

# So that we at least contain the artifact in the window (minimum 2ms),
# we attempt to start window s.t. stim begins on average 1ms in.
# If stim arrives 1ms late, it will be marked in the next 2ms window (but this shouldn't affect 5ms analyses)
# 600us

_CONDITION_KEY = 'condition'
_MULTISESSION_CONDITION_KEY = 'session_condition'
_SENSORY_ARRAY_KEY = 'stim' # kind of legacy mistake.

def argsorted(seq):
    # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)
class ICMSDataset(Dataset):
    r"""
        Monolith class managing raw spikes and trial metadata.

        Note on `key_df`:
            - This dataframe tracks trial metadata and should be synced to `record_states` and `stim_states` at all times.
            - It's not entirely clear why I don't just use the dataframe entirely - perhaps because I'm uncomfortable with the efficiency of tensor ops on dfs? (Not too rational, since __getitem__ does individual indexing.)
    """
    def __init__(
        self,
        config: CN,
        override_cache=False,
        override_path: str=None,
        should_trim=True,
    ):
        super().__init__()
        self.full_cfg = config
        self.config = config.DATASET
        self.seed = self.config.SPLIT_SEED # Used in subsetting procedures

        if override_path:
            self.config.defrost()
            self.config.FILE = override_path
            self.config.MULTISESSION = []
            self.config.freeze()

        if getattr(self.config, 'MULTISESSION', []):
            if len(self.config.MULTISESSION) < 2:
                raise ValueError("MULTISESSION must be a list of at least 2 sessions.")
            self.config.defrost()
            self.config.MULTISESSION = (np.array(self.config.MULTISESSION)[argsorted(self.session)]).tolist()
            self.config.freeze()
            self.dataset_path = [Path(self.config.DIR) / e for e in self.config.MULTISESSION]
        else:
            self.dataset_path = list(Path(self.config.DIR).glob(self.config.FILE))
        if len(self.dataset_path) > 1:
            assert self.config.STYLE == 'raw_stim_record', f"merging only implemented for {self.config.STYLE}"
        else:
            self.dataset_path = self.dataset_path[0]

        # 1-indexed
        self.record_channels = None
        self.stim_channels = None
        self.session_idx_map = None

        self.aligned = False
        self.load_trialized_states(override_cache)
        self.trimmed = False
        should_trim = should_trim and not self.multisession # don't trim for multisession, ever
        if should_trim: # definitely trim for training
            self.stim_states, self.record_states = self.trim_trials(
                self.stim_states,
                self.record_states,
                should_align_to_first_stim=self.config.STYLE == 'quicklogger' # Temp for new raw S1 data, in general this should be true
            )
            # crop to max length
            if self.multisession:
                self.stim_states = self.stim_states[:,:self.config.MAX_TRIAL_LENGTH]
                self.record_states = self.record_states[:,:self.config.MAX_TRIAL_LENGTH]
        else:
            if self.multisession:
                self.stim_states = [s[:self.config.MAX_TRIAL_LENGTH] for s in self.stim_states]
                self.record_states = [s[:self.config.MAX_TRIAL_LENGTH] for s in self.record_states]
        if self.config.DISTILL_TARGET:
            self.teacher_states = load_distilled_rates(self.config.DISTILL_TARGET, self)
            print(f"Distilling from {self.config.DISTILL_TARGET}, make sure the teacher was trained on the same/correct dataset...")
            assert self.record_states.size()[:-1] == self.teacher_states.size(), "Mismatch in distill target size"
        else:
            self.teacher_states = None

    def stitch_dfs(self, dfs: Dict[SessionSet, pd.DataFrame]) -> pd.DataFrame:
        for session in dfs:
            dfs[session]['session'] = [(session)] * len(dfs[session])
            dfs[session]['session_idx'] = self.session.index(session) # ! Assumes we're not subsetting into a different order
            dfs[session]['unique'] = dfs[session].apply(lambda x: f'{x[self.config.UNIQUE_KEY]}-{session}', axis=1) # ! I don't actually think this is necessary but...
            dfs[session][_MULTISESSION_CONDITION_KEY] = dfs[session].apply(
                lambda x: session[0] * 50000 + session[1] * 2000 + x.condition, axis=1 # no more than 2000 conditions per set (it's the trial limit), no more than 25 sets per session (per `session_info`)
            ) # a unique, session specific condition hash.
            # since we don't expect to be merging across conditions
        return pd.concat(dfs).reset_index()

    def load_trialized_states(self, override_cache=False):
        assert self.config.STYLE in ['quicklogger', 'raw_stim_record']
        self.subsetted = False # for train val
        self.subsetted_stim = False
        self.subsetted_record = False
        self.aligned = False

        # Each sub-implementation needs to load states, channels, metadata df, and unique keys for that df.
        if self.config.STYLE == 'quicklogger':
            r"""
                Quicklogger event files are always be aligned to first stim onset.
                For now, we will assume that the blanked 1ms are in the same bin as stim onset
                # TODO; However this is not generally true, and we may want to either
                1. Manually flag the blanked bins
                2. Let the model know when the stim timing was, within bin. (and include blank prediction in loss)
            """
            with open(self.dataset_path, 'rb') as f:
                all_events = pickle.load(f)
            df = pd.DataFrame(all_events) # Load df so that we can subselect relevant trials for train and test
            self.key_df = df[[self.config.SPLIT_KEY]]
            assert False, "This codepath is old and likely buggy!"
        elif self.config.STYLE == 'raw_stim_record':
            def load_df(path, full=True):
                data = torch.load(path)
                # from B C T to B T C 1
                # self.record_states = torch.tensor(data['spikes']).permute(0, 2, 1).unsqueeze(-1)
                payload = {
                    'record': data['spikes'],
                    'trial_num': data['trial_num'],
                    'stim_time': data['stim_time'], # TODO sub with recorded NEV stim
                }
                if full:
                    # ! careful about composition order?
                    payload = {**payload,
                        'src_file': data['src_file'],
                        _CONDITION_KEY: data['condition'],
                        'command_time': [c[0] for c in data['command']], # tuple (time, channel (1-64), amp)
                        'command_channels': [c[1] for c in data['command']], # tuple (time, channel (1-64), amp)
                        'command_current': [c[2] for c in data['command']], # tuple (time, channel (1-64), amp)
                    }
                    if 'stim_dir' in data and not data.get('ignore_exp_info', False):
                        exp_info = Path(data['stim_dir']) / 'exp_info.pkl'
                        if exp_info.exists():
                            with open(exp_info, 'rb') as f:
                                exp_info: Dict[TrialNum, Dict[MetadataKey, Any]] = pickle.load(f)
                            for key in exp_info[1]: # 1 is arbitrary
                                payload[key] = [exp_info[t][key] for t in payload['trial_num']]
                    if 'channels' not in payload:
                        if 'channel' in payload: # legacy TODO convert to tuples
                            payload['channels'] = payload['channel']
                            del payload['channel']
                        else:
                            print("Inferring channel metadata from command")
                            payload['channels'] = [tuple(np.unique(cc)) for cc in payload['command_channels']] # ! STIM CHANNELS
                    if 'protocol' not in payload:
                        print("Inferring protocol metadata from command")
                        def infer_protocol(cc):
                            if len(np.unique(cc)) <= 2: # (0 and) command current
                                return "standard"
                            return "rap"
                        payload['protocol'] = [infer_protocol(cc) for cc in payload['command_current']]
                    # protocol is only needed for session 48...
                    if self.config.SUBSET_PROTOCOL == "":
                        del payload['protocol']
                df = pd.DataFrame(payload)
                del payload
                df.sort_values(by='trial_num', inplace=True) # ! Needed to ensure tensors are built in trial_num order. Should be true, but just in case...
                return df
            if isinstance(self.dataset_path, list):
                dfs = {s: load_df(p, full=True) for s, p in zip(self.session, self.dataset_path)}
                self.key_df = self.stitch_dfs(dfs)
            else:
                self.key_df = load_df(self.dataset_path)
        # import pdb;pdb.set_trace()
        for subject in subject_to_array: # assume single-subject
            if subject in str(self.key_df.iloc[0].src_file):
                self.subject = subject
                self.array_info = subject_to_array[subject]()
        if override_cache:
            loaded = False
        else:
            loaded = self._load_cache_if_exists()
        if not loaded:
            if self.config.STYLE == 'quicklogger':
                self.compute_states_from_events(df, all_events, cache=True)
            elif self.config.STYLE == 'raw_stim_record':
                # TODO if implementing fragmenting, track preprocessed files here...
                self.compute_states_from_trials()
                # following columns are tracked in self.record_states, self.stim_states
            self._save_cache()
        # import pdb;pdb.set_trace()
        self.key_df = self.key_df.drop(['record', 'stim_time'], axis=1) # not gc-ed
        self.subset_record_channels(**self.get_array_config())

    def get_array_config(self):
        pedestals = []
        if "lateral" in self.config.RECORD_ARRAY:
            pedestals = [0]
        elif "medial" in self.config.RECORD_ARRAY:
            pedestals = [1]
        return {
            "stim_only": "sensory" in self.config.RECORD_ARRAY or self.config.RECORD_ARRAY == _SENSORY_ARRAY_KEY, # latter is legacy
            "pedestals": pedestals,
            "channels": getattr(self.config, 'RECORD_CHANNELS', [])
        }

    @staticmethod
    def _split_trials(df, trial_ratio=0.5, seen=True):
        r"""
            Manual split for scaling. First half is seen/for training, Second half is unseen/for testing.
        """
        if seen:
            return df[df.index < len(df) * trial_ratio]
        return df[df.index >= len(df) * trial_ratio]

    def _load_cache_if_exists(self):
        # cache contains processed, binned timeseries data ready to model
        cache_path = self._cache_path()
        if not osp.exists(cache_path):
            return False
        # Caching is invariant across prefix settings, but sensitive to bin size.
        cache = torch.load(cache_path, map_location='cpu')
        self.stim_states = cache["stim"]
        self.record_states = cache["record"]
        self.stim_channels = cache["stim_channels"]
        self.record_channels = cache["record_channels"]
        print(f"Trialized data loaded from {cache_path}.")
        return True

    def _save_cache(self):
        if self.multisession: # no caching in this case
            return
        # TODO implement session-wise caching
        cache_path = self._cache_path()
        torch.save({
            "stim": self.stim_states,
            "record": self.record_states,
            "stim_channels": self.stim_channels,
            "record_channels": self.record_channels,
        }, cache_path)
        print(f"Trialized data computed and cached at {cache_path}")

    def hash_session_file(self, f):
        return osp.splitext(f)[0]

    @property
    def cache_hash(self):
        r"""
            hash for dataset files
        """
        if self.multisession:
            roots = [self.hash_session_file(f) for f in self.config.MULTISESSION]
            return "-".join(roots)
        else:
            return self.hash_session_file(self.config.FILE)

    def _cache_path(self):
        if 'prefix' in self.config.CACHE:
            cache_path = "{file}-{bin_size}.cache" # deprecated
        else:
            cache_path = self.config.CACHE
        return osp.join(self.config.DIR, cache_path.format(
            file=self.cache_hash,
            bin_size=self.config.TIME_BIN_SIZE
        ))

    def compute_states_from_trials(self):
        self.stim_channels = np.arange(64, dtype=int) + 1 # This is manual, nothing explicitly declares this, indeed, it might as well be size 1-4 for all the channels we use...
        self.stim_states = []
        self.record_states = []
        for trial_record, trial_stim_samples, command_time, command_channel, command_current in zip(
            self.key_df['record'],
            self.key_df['stim_time'],
            self.key_df['command_time'],
            self.key_df['command_channels'],
            self.key_df['command_current']
        ):
            if self.record_channels is None:
                self.record_channels = np.arange(trial_record.shape[0], dtype=int) + 1
            trial_record = torch.tensor(trial_record).T.unsqueeze(-1) # C T to T C 1
            # Target shape: Trial[time bin x channel x feat]
            # TODO replace with NEV
            r""" EXTRACT STIMULATION PARAMETERS
                Our preprocessing will extract putative stimulation times (`trial_stim_samples`),
                which we need to identify the true stimulation parameters for (such as channel, amplitude).
                We do need to compare timing as some (small but non-negligible) fraction of pulses are dropped.
                TODO: the most technically correct way of doing this is to extract from voltage monitor, i.e. hardware logs right before stimulation
                - We will NEED to use voltage monitor once using multi-channel stimulation.
                Right now, we're referring to the commanded parameters, assuming minimal drift and no egregious dropping (i.e. not robust to severe jitter and assuming each command has an artifact.)

                Note we only use parameters, and not timing, as command timing clearly drifts a ~2ms.
                Assuming effected stim conserves command order, we'll zip backwards.
                i.e. assuming last pulse was effected, then take next detected stim's params from the closest of remaining commands
                # In simplest case, this turns into taking last N
            """
            def simple_backzip(command_time, effected_time):
                effected_time = effected_time + (command_time[-1] - effected_time[-1])
                effected_command_indices = [] # ! Not all commanded are effected (dropped pulses).
                commanded_effect_indices = [] # ! Not all detected artifacts are true artifacts (i.e. transient). While the model should know about this, we're currently just dropping them.
                cmd_i = len(command_time) - 1
                effect_i = len(effected_time) - 1
                while effect_i >= 0 and cmd_i >= 0:
                    # assuming alignment at current timestep, a few possible scenarios:
                    # either next effect is closer to cmd
                    if cmd_i == 0 or \
                        abs(command_time[cmd_i] - effected_time[effect_i]) <= abs(command_time[cmd_i-1] - effected_time[effect_i]):
                        # In this case, the current effected time likely matches the current commanded time, but it may not be the best fitting effected time
                        # (Sometimes we have multiple mini-artifax per command)
                        # Sweep back for the best one. In particular, "best" is history biased since
                        while (effect_i > 0 and \
                            abs(command_time[cmd_i] - effected_time[effect_i - 1]) <= abs(command_time[cmd_i-1] - effected_time[effect_i - 1])): # As long as effect_i is more likely to belong to current cmd than previous one
                            # abs(command_time[cmd_i] - effected_time[effect_i]) >= abs(command_time[cmd_i] - effected_time[effect_i-1])):
                                effect_i -= 1
                        effected_command_indices.append(cmd_i)
                        commanded_effect_indices.append(effect_i)
                        # print(f"Cmd: {cmd_i}/{len(command_time)} \t Effect: {effect_i}/{len(effected_time)} \t {command_time[cmd_i]}:{effected_time[effect_i]:.4f}")
                        # Adjust effected time labels for purpose of matchign with command. (Same as moving command, probably more legible that way.)
                        # This presumes (and Jeff confirms) stimulator attempts to maintain diffs rather than absolute command times (dropping commands when infeasible)
                        effected_time = effected_time + command_time[cmd_i] - effected_time[effect_i] # careful not to mutate true
                        effect_i -= 1
                        # In order to account for monotonic drift, we need to constantly re-align effected time to zipped command time
                    cmd_i -= 1
                return list(reversed(effected_command_indices)), list(reversed(commanded_effect_indices))
            fs = 30000
            effected_idx, commanded_idx = simple_backzip(command_time, trial_stim_samples / fs)
            # aligned_command_time = command_time + trial_stim_samples[-1] / fs - command_time[-1]

            channels = torch.tensor(command_channel[effected_idx] - 1).long()
            currents = self.transform_current(torch.tensor(command_current[effected_idx]), normalize=True)
            # While true channel and current info come from command (as far as we know), best timing info should be taken from effected stim itself
            # ! We are discarding the other artifact fragment markers by only reporting true commands here. Consider denoting other artifact times.
            trial_stim_samples = torch.tensor(trial_stim_samples[commanded_idx].astype(int))
            trial_stim_state = torch.zeros(
                trial_record.size(0),
                len(self.stim_channels),
                # 1 # count
                3 # count, current, timing
            ).float()
            bin_samples = int(self.config.TIME_BIN_SIZE * fs)
            timebins = torch.div(trial_stim_samples, bin_samples, rounding_mode='floor').long() # just rounded, add timing cf truncation in preprocessing
            try:
                trial_stim_state[timebins, channels, 0] = 1.
                trial_stim_state[timebins, channels, 1] = currents
                trial_stim_state[timebins, channels, 2] = (trial_stim_samples % bin_samples) / bin_samples - 0.5
            except:
                # Somehow, there are some strange trials where stim is just not recorded properly e.g. only 5 pulses in 1s...?
                print("Invalid stim attempted, skipping trial.")
                # TODO delete invalid trials from key_df or mark them somehow
            self.record_states.append(trial_record)
            self.stim_states.append(trial_stim_state)
        # TODO clean up df

    @staticmethod
    def transform_current(current, normalize=False):
        # approx z-score.
        if normalize:
            return current / 100 -0.5
        else:
            return (current + 0.5 * 100)

    def compute_states_from_events(self, df, all_events, cache=True):
        # Compute dataset-level organizing structure
        def get_unique_channels_and_maps(channels, pedestals=None, blacklist=[]) -> Tuple[np.ndarray, Dict[np.uint8, int]]:
            # args
            #   channels: list of all channels (un-offset, as recorded in Central)
            #   pedestal: bank index paired with `channels`
            #   blacklist: (channel, pedestal) tuples to exclude
            # returns:
            #   unique_channels: to count # of slots we should assign
            #   channel_map: maps of channel key into slots
            # TODO support multi-channel by flattening before unique call

            if pedestals is None:
                unique_channels, src_indices = np.unique(channels, return_index=True)
            else:
                unique_channels, src_indices = np.unique(
                    np.stack([channels, pedestals], axis=1), # N x 2,
                    return_index=True,
                    axis=0
                ) # return unique channel pedestal pairs
            # Note, there's some casting here, be careful
            if blacklist:
                assert pedestals is not None, "Shouldn't be blacklisting on channel only yet"
                filt_channels = []
                filt_indices = []
                for i in range(len(unique_channels)):
                    if tuple(unique_channels[i]) not in blacklist:
                        filt_channels.append(unique_channels[i])
                        filt_indices.append(src_indices[i])

                unique_channels = np.array(filt_channels)
                src_indices = filt_indices

            id_to_unique_map = defaultdict(lambda: -1) # * blacklist to -1, which will later be filtered during collection
            for i, ind in enumerate(src_indices):
                el = (channels[ind], None if pedestals is None else pedestals[ind]) # Use regular tuples instead of unhashable np arrays
                id_to_unique_map[el] = i
            return unique_channels, id_to_unique_map

        # Channels will be slotted in dataset tensor in order they appear in `unique` return
        self.stim_channels, stim_channel_map = get_unique_channels_and_maps(df['metadata_channel'])
        record_channels_seen = np.concatenate(df['spikes_channel'])
        record_channels_pedestal = np.concatenate(df['spikes_source_index']) # NSP

        blacklist_channels, blacklist_pedestals = self.array_info.get_blacklist_channels(flatten=False)
        blacklist_tuples = list(zip(blacklist_channels, blacklist_pedestals)) # cast to list for debuggability (zip vanishes after iteration)
        self.record_channels, record_channel_map = get_unique_channels_and_maps(
            record_channels_seen, record_channels_pedestal, blacklist=blacklist_tuples
        )
        assert False, "old codepath, check typing on record_channels and stim_channels"

        self.stim_states, self.record_states = self.events_to_timeseries(
            all_events,
            stim_channel_map=stim_channel_map,
            record_channel_map=record_channel_map,
        )


    @staticmethod
    def get_mean_and_std(states):
        if states is None:
            return None
        if isinstance(states, list):
            print("Warn: Model not initialized with mean and std from data. OK if loading checkpoint.")
            return None
        # states BxTxC
        flat = states.flatten(0, 1).float()
        return flat.mean(0), flat.std(0)

    @property
    def pristine(self) -> bool:
        # ! Be very careful... I'm allowing record subset in cache because we mostly care about sensory subsets right now... and we shouldn't fail silently once we re-enable more channels due to shape mismatch
        # The "solution" is to cache with hash of record channels, stim channels, and trials.
        return not (self.subsetted or self.subsetted_stim)
        # return not (self.subsetted or self.subsetted_stim or self.subsetted_record)

    @property
    def unique_key(self):
        if self.multisession:
            return 'unique' # see `stitch_dfs`
        return self.config.UNIQUE_KEY

    @property
    def unique_keys(self) -> np.ndarray:
        # return copies to prevent mutation (i.e. from shuffling)
        return self.key_df[self.unique_key].unique().copy()

    @property
    def split_keys(self) -> np.ndarray:
        return self.key_df[self.config.SPLIT_KEY].unique().copy()

    @property
    def multisession(self) -> bool:
        return bool(self.config.MULTISESSION)

    @staticmethod
    def extract_session_num(file) -> SessionSet:
        return int(re.match('(\d{5})\.', file).group(1)), \
            int(re.match('.*\.Set(\d{4})\.', file).group(1))

    @staticmethod
    def extract_session_nums(files: Union[str, List[str]]) -> Union[SessionSet, Tuple[SessionSet, ...]]:
        if isinstance(files, str):
            return ICMSDataset.extract_session_num(files)
        ret = tuple(ICMSDataset.extract_session_num(f) for f in files)
        return sorted(ret)

    @staticmethod
    def extract_session_num_from_cfg(cfg: CN) -> Union[SessionSet, Tuple[SessionSet, ...]]:
        if cfg.MULTISESSION:
            return ICMSDataset.extract_session_nums(cfg.MULTISESSION)
        return ICMSDataset.extract_session_nums(cfg.FILE)

    @property
    def session(self) -> Union[SessionSet, Tuple[SessionSet, ...]]:
        return self.extract_session_num_from_cfg(self.config)

    @property
    def session_num(self) -> int:
        if isinstance(self.session[0], int):
            return self.session[0]
        return tuple(s[0] for s in self.session)

    @property
    def _condition_key(self) -> str:
        if self.multisession:
            return _MULTISESSION_CONDITION_KEY
        return _CONDITION_KEY

    @property
    def record_events(self):
        return self.record_states[..., 0]

    @property
    def stim_events(self):
        return self.stim_states[..., 0]

    @property
    def bin_size(self) -> float:
        # TODO import from dataset creation metadata, if data arrives binned.
        # As is, we need to set it...
        return self.config.TIME_BIN_SIZE

    def get_attrs(self):
        r"""
        # Provide dataset info to shape model
        Note in multisession scenarios, we generally assume that datasets are of the same shape (so we don't necessarily need task-specific inputs)
        """
        return DatasetAttrs(
            stim_channels=self.stim_channels,
            record_channels=self.record_channels,
            record_mean_std=self.get_mean_and_std(self.record_states), # For normalizing
            teacher_mean_std=self.get_mean_and_std(self.teacher_states),
            stim_channel_dim=self.stim_states[0].size(-1),
            subject=self.subject,
            record_array=self.config.RECORD_ARRAY, # ! IMPLEMENT
            bin_size=self.bin_size, # anti-pattern, but convenient for batched cropping in model
        )

    # =====
    # Subsetting, dataset manipulation
    # =====

    # ===== Batch/trial subsetting =====
    # ! Give a copy, not the reference to the keys which can be in place shuffled...

    def get_key_indices(self, key_values, key=None):
        key = key or self.unique_key
        return copy.deepcopy(self.key_df[self.key_df[key].isin(key_values)].index)

    def subset_by_key(self, key_values: List[Any], key=None, allow_second_subset=True):
        r"""
            # ! In place
            # (Minimum generalization is, same trial, different pulse)
            # To new trials, likely practical minimum
            # To new channel, amplitudes, etc. in the future
            Note - does not update `self.session` as we want to track the source (multi-)sessions to prevent unintended session mixup (see `merge`)
        """
        if self.subsetted:
            assert allow_second_subset
            print()
            print("Warning!!! Dataset has already been subsetted") # Subsetting multiple times is _really_ untested.
            print()
        if len(key_values) == 0:
            print("Info: No keys provided, ignoring subset.")
            return
        key = key or self.unique_key
        key_indices = self.get_key_indices(key_values, key=key)
        self.key_df = self.key_df[self.key_df[key].isin(key_values)]
        # ! Making the choice to forget old trial information
        self.key_df = self.key_df.reset_index(drop=True)
        if self.trimmed:
            self.record_states = self.record_states[key_indices]
            self.stim_states = self.stim_states[key_indices]
            if self.teacher_states is not None:
                self.teacher_states = self.teacher_states[key_indices]
        else:
            self.record_states = [self.record_states[k] for k in key_indices]
            self.stim_states = [self.stim_states[k] for k in key_indices]
            if self.teacher_states is not None:
                self.teacher_states = [self.teacher_states[k] for k in key_indices]
        self.subsetted = True
        return key_indices

    def get_conditions(self) -> ConditionInfo:
        r"""
            returns:
                Dict[condition key, [trial idxes]]. Idxes might be literal integer indices, or trial nums, uncertain.
        """
        if self._condition_key not in self.key_df:
            # condition is in all the datasets JY prepared. Only other investigated data is survey data.
            conditions = PSTHDataset.id_conditions_for_survey(self.dataset.stim_states)
            trial_conds = {}
            for c in conditions:
                for trial in conditions[c]:
                    trial_conds[trial] = c
            self.key_df[self._condition_key] = self.key_df.apply(lambda x: trial_conds[x.trial_num])
        return self.key_df.groupby(self._condition_key).groups

    def get_unique_conditions(self) -> List[Condition]:
        return self.get_conditions().keys()

    def get_condition_trials(self, condition):
        trials = self.key_df[self.key_df[self._condition_key] == condition].index
        return zip(*[self[t] for t in trials])

    def subset_by_condition(self, conditions):
        if self.multisession:
            print("Warn: subsetting by condition explicitly in multisession. Did you manually hash the desired conditions?")
        self.subset_by_key(conditions, key=self._condition_key)

    def subset_scaling(self, heldin=True):
        r"""
            If heldin, return the heldin trials, else return the heldout trials.

            First compute the heldout set,
            Then subset heldin to desired scaling length (calibration/intra-scaling amount).
        """
        np.random.seed(self.seed)
        subset_df = self.key_df
        if getattr(self.config, 'SCALING_HOLDOUT_SESSION', 0):
            if self.multisession:
                subset_df = subset_df[subset_df.session == (self.config.SCALING_HOLDOUT_SESSION, self.config.SCALING_HOLDOUT_SESSION_SET)]
            else:
                assert self.session == (self.config.SCALING_HOLDOUT_SESSION, self.config.SCALING_HOLDOUT_SESSION_SET)
        holdout_amount = int(len(subset_df) * self.config.SCALING_HOLDOUT_FRACTION)
        holdout_trials = np.random.choice(subset_df[self.unique_key], holdout_amount, replace=False)
        if heldin:
            if self.config.SUBSET_TRIAL_LIMIT:
                intra_holdin = np.setdiff1d(subset_df[self.unique_key], holdout_trials)
                intra_holdin = np.random.choice(intra_holdin, self.config.SUBSET_TRIAL_LIMIT, replace=False)
                holdout_trials = np.setdiff1d(subset_df[self.unique_key], intra_holdin)
            holdin_trials = [t for t in self.key_df[self.unique_key] if t not in holdout_trials]
            self.subset_by_key(holdin_trials)
        else:
            self.subset_by_key(holdout_trials)

    def get_configured_protocol_conditions(self, protocols: List[str]=[], fraction_in: float = 0.8):
        r"""
            Identify, per protocol, to a fraction subset of total conditions in protocol.
            If `protocols` is not specified, go through all available protocols.
            Used for evaluating transfer across conditions within a single dataset.

            returns:
                conditions that compose the specified subset of available protocols
                (and complement, for N-shot)
        """
        # TODO maybe switch to stratified split instead of homebrewed thing here.
        _fraction = fraction_in
        # _fraction = self.config.SUBSET_FRACTION_CONDITIONS_IN_PROTOCOL
        if _fraction == 0 or _fraction == 1:
            print("Not subsetting within protocol.")
            return [], [] # Degenerate, no op
        if not protocols:
            protocols = self.key_df['protocol']
        protocol_conditions = {}
        for p in protocols:
            protocol_conditions[p] = self.key_df[self.key_df['protocol'] == p][self._condition_key].unique()
        included_conditions = []
        excluded_conditions = [] # excluded, in protocol
        for protocol in protocol_conditions:
            np.random.seed(self.seed) # ALWAYS RANDOM SEED SO WE GET THE SAME DRAW REGARDLESS OF HOW MANY TIMES WE DO IT (consistent across protocol splits)
            included = np.random.choice(
                protocol_conditions[protocol],
                size=(math.floor(len(protocol_conditions[protocol]) * _fraction),),
                replace=False
            )
            included_conditions.extend(included)
            excluded_conditions.extend(list(set(protocol_conditions[protocol]).difference(included)))
        return included_conditions, excluded_conditions

    def subset_configured_protocol_conditions(self, protocols: List[str] = []):
        config_conditions, _ = self.get_configured_protocol_conditions(protocols, self.config.SUBSET_FRACTION_CONDITIONS_IN_PROTOCOL)
        if config_conditions:
            print(f"\n Using configured conditions: {config_conditions}\n Subsetting...")
            self.subset_by_key(config_conditions, key=self._condition_key)

    def exclude_configured_protocol_conditions(self, protocols: List[str] = []):
        # Complement of above function
        _, config_conditions = self.get_configured_protocol_conditions(protocols, self.config.SUBSET_FRACTION_CONDITIONS_IN_PROTOCOL)
        if config_conditions:
            print(f"\n Excluding configured conditions, remaining: {config_conditions}\n Subsetting...")
            self.subset_by_key(config_conditions, key=self._condition_key)


    def tv_split_by_split_key(self, train_ratio=0.8, seed=None):
        keys = copy.deepcopy(self.split_keys)
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        np.random.shuffle(keys)
        tv_cut = int(train_ratio * len(keys))
        train_keys, val_keys = keys[:tv_cut], keys[tv_cut:]
        return train_keys, val_keys

    # ! Shuffling assumes our dataset is balanced
    # Keys determine how we split up our dataset. By trial, or more specific ICMS parameter
    def create_tv_datasets(self, **kwargs):
        train_keys, val_keys = self.tv_split_by_split_key(**kwargs)
        train = copy.deepcopy(self)
        train.subset_by_key(train_keys, key=self.config.SPLIT_KEY)
        val = copy.deepcopy(self)
        val.subset_by_key(val_keys, key=self.config.SPLIT_KEY)
        return train, val

    def merge(self, data_other: Self):
        assert data_other.session == self.session, "Only support merging back of subsetted dataset (for N-shot) right now"
        self.key_df = pd.concat([self.key_df, data_other.key_df])
        self.key_df = self.key_df.reset_index(drop=True)
        self.record_states = torch.cat([self.record_states, data_other.record_states], 0)
        self.stim_states = torch.cat([self.stim_states, data_other.stim_states], 0)
        if self.teacher_states is not None:
            self.teacher_states = torch.cat([self.teacher_states, data_other.teacher_states], 0)

    def create_n_shot_experiment(
        self,
        holdin_conditions=None,
        holdout_conditions=None,
        compute_holdout=False
    ) -> Tuple[Self, Self, Union[Self, None]]:
        r"""
            Set up n shot experiment. Returns train/val that do primarily consist of held in conditions.
            holdin_conditions, holdout_conditions: if not None, use as respective initial splits.
                Does not affect subsequent train/val/shot injection logic.
                For now, must be provided together (probably doesn't need to be.)
            compute_holdout: if True, returns always held out trials for held out conditions. Used during evaluation only.
        """
        print(f"Generating N-shot exp. holdin: {holdin_conditions}, holdout: {holdout_conditions}")
        np.random.seed(self.config.SPLIT_SEED)
        assert (holdin_conditions is None and holdout_conditions is None) or (
            holdin_conditions is not None and holdout_conditions is not None
        )
        holdin = copy.deepcopy(self)
        if holdin_conditions is None or holdout_conditions is None:
            keys = self.split_keys
            np.random.shuffle(keys)
            out_cut = int(self.config.N_SHOT_HOLDOUT_FRACTION * len(keys))
            holdout_keys, holdin_keys = keys[:out_cut], keys[out_cut:]
            if getattr(self.config, 'N_SHOT_CUT_HOLDIN', 1.0) < 1.0:
                keep = round(len(holdin_keys) * self.config.N_SHOT_CUT_HOLDIN)
                if keep == 0:
                    holdin_keys = []
                else:
                    holdin_keys = holdin_keys[:keep]
            holdin.subset_by_key(holdin_keys, key=self.config.SPLIT_KEY, allow_second_subset=False)
        else:
            holdin.subset_by_condition(holdin_conditions)

        train, val = holdin, copy.deepcopy(holdin)
        keys = holdin.unique_keys
        np.random.shuffle(keys)
        tv_cut = int(0.8 * len(keys))
        train_keys, val_keys = keys[:tv_cut], keys[tv_cut:]
        train.subset_by_key(train_keys)
        val.subset_by_key(val_keys)
        holdout = None
        if compute_holdout or self.config.N_SHOT_TRAIN_FRACTION > 0:
            holdout = copy.deepcopy(self)
            if holdout_conditions is not None:
                holdout.subset_by_condition(holdout_conditions)
            else:
                holdout.subset_by_key(holdout_keys, key=self.config.SPLIT_KEY, allow_second_subset=False)

            holdout_keys = []
            conditions = []
            if self.config.HOLDOUT_GOLDEN_CONDITION:
                holdout_conditions = holdout.get_conditions()
                train_conditions, golden_conditions = train_test_split(
                    list(holdout_conditions.keys()), test_size=self.config.N_SHOT_GOLDEN_FRACTION,
                    random_state=self.config.SPLIT_SEED
                )
                train_shots = []
                golden_shots = []
                _tc = []
                _gc = []
                for c in golden_conditions:
                    golden_shots.extend([holdout.key_df.iloc[t][self.config.UNIQUE_KEY] for t in holdout_conditions[c]])
                    _gc.extend([c] * len(holdout_conditions[c]))
                for c in train_conditions:
                    train_shots.extend([holdout.key_df.iloc[t][self.config.UNIQUE_KEY] for t in holdout_conditions[c]])
                    _tc.extend([c] * len(holdout_conditions[c]))
                train_conditions = _tc
                golden_conditions = _gc
            else:
                # Do your best to get a balanced split of these holdout keys
                for cond, trial_idxes in holdout.get_conditions().items():
                    holdout_keys.extend([holdout.key_df.iloc[t][self.config.UNIQUE_KEY] for t in trial_idxes])
                    conditions.extend([cond] * len(trial_idxes))
                train_shots, golden_shots, train_conditions, golden_conditions = train_test_split(
                    holdout_keys, conditions,
                    test_size=self.config.N_SHOT_GOLDEN_FRACTION,
                    random_state=self.config.SPLIT_SEED,
                    stratify=conditions
                )
            if self.config.N_SHOT_TRAIN_FRACTION > 0:
                if self.config.N_SHOT_TRAIN_FRACTION < 1:
                    try:
                        train_shots, discarded_shots = train_test_split(
                            train_shots, train_size=self.config.N_SHOT_TRAIN_FRACTION, stratify=train_conditions
                        )
                    except:
                        # Due to data pipeline leakiness, we sometimes end up with conditions with only one trial. In this case, don't worry about balanced exposure to held-in conditions (just the quickest fix here)
                        # The net effect is that the % of novel condition we are claiming becomes only an average claim across conditions, rather than per condition.
                        print("Warn: Not enough trials to stratify exposure to novel conditions. Falling back to random split.")
                        train_shots, discarded_shots = train_test_split(
                            train_shots, train_size=self.config.N_SHOT_TRAIN_FRACTION
                        )
                merge_payload = copy.deepcopy(holdout)
                merge_payload.subset_by_key(train_shots)
                # reduce train by merge_payload size to hold train set size constant
                train_keys = train.unique_keys
                np.random.shuffle(train_keys)
                train_cut = len(train_keys) - len(merge_payload)
                train_keys = train_keys[:train_cut]
                train.subset_by_key(train_keys)
                train.merge(merge_payload)

            if compute_holdout:
                holdout.subset_by_key(golden_shots)
        return train, val, holdout

    # ===== Channel subsetting =====

    def subset_record_channels(self, channels=[], pedestals=[], stim_only=False):
        r"""
            Subset record channels. Will _always_ remove blacklisted channels.
            Note that since this is fairly unrreversible, we constrain it to be only run once.
            If `channels` or `pedestal` or `stim_only` is specified, will additionally remove those channels.
            # FYI I think this is only tested to work when only one of the above options is true.
        """
        assert not self.subsetted_record, "cannot subset record twice, please reset before using"
        if not channels: # channels overrides all if provided
            channels = self.array_info.get_subset_channels(
                pedestals=pedestals,
                use_motor=not stim_only,
                use_sensory=True
            )
        channels = np.setdiff1d(channels, self.array_info.get_blacklist_channels())

        if isinstance(self.record_states, torch.Tensor):
            self.record_states = self.record_states[:,:, channels - 1]
        else:
            self.record_states = [t[:, channels - 1] for t in self.record_states]
        self.record_channels = channels
        self.subsetted_record = True

    # ! The following is implemented as a trial subsetting method, not a channel subsetting method.
    # ! To avoid de-syncing with df channel information, don't actually update stim state channel indices - just exclude trials that use blacklisted channels (at any point).
    def subset_stim_channels(self, channels=None, pedestal=None):
        assert not self.subsetted_stim, "cannot subset stim twice, reset"
        assert channels is not None or pedestal is not None, "need subset args"
        if pedestal == 0:
            channels = np.arange(32, dtype=int) + 1
            self.stim_channels = channels
        elif pedestal == 1:
            channels = np.arange(32, dtype=int) + 32 + 1
            self.stim_channels = channels
        # Update df, record states, stim states _by_ trials
        self.subset_by_key(channels, key='channel') # TODO needs update if we do multi-pedestal stim... which we won't.
        self.subsetted_stim = True

    def events_to_timeseries(
        self,
        events, # Events are trial windows, actually, not like, spike events
        stim_channel_map,
        record_channel_map,
    ):
        """
            For now, I'm not building a global timeline, since I only need to immediately chop afterwards, and for the foreseeable future I'll be aligning to first stim
            I'm simply going to prep tensorized data
            (Sorry for misleading function name)
            TODO we may want to jitter first stim time else model will overfit to predicting at step T.

            returns:
                record_states: list (len B, trial) of [Time, Channel (rec channels), N_rec=1 (data per channel, mostly for symmetry with stim, may e.g. include fraction blanked)] # TODO what I e.g.-ed
                stim_states: list (len B, trial) of [Time, Channel (stim channels), N_stim (data per channel)]
                - Aligned most probably to a fixed prefix from first stim, otherwise, time of first record spike
                - B: trial should be a stim train, fortunately equiv to an experimental trial
                - Time: relative to `timeseries_start`, in `resolution` bins
                - Channel: Stim/recording channel (distinct for now...)
                - Data:
                    - 0/1 flag if there was stim
                    - normalized float for amplitude (can collapse with ^ by saying 0)
                    - Optional waveform encoding (biphasic timing) # TODO
        """

        record_states = []
        stim_states = []
        print("Building timeseries...")
        valid_stim = len(stim_channel_map)
        valid_record = len(record_channel_map)
        for event in events: # map to T C N
            end_time = max(
                event["sync_source_timestamp"][-1],
                *event["spikes_source_timestamp"]
            )

            # Create bins aligned to stim start time so that stim are ~consistently timed within a bin across trials
            start_time = event["sync_source_timestamp"][0] - MAX_PREFIX_BUFFER
            # start_time = max(
            #     min(*event["spikes_source_timestamp"]),
            #     event["sync_source_timestamp"][0] - MAX_PREFIX_BUFFER
            # )
            bins = np.arange(
                start_time - TIME_EPSILON - SPIKE_TIMING_OFFSET,
                end_time + self.config.TIME_BIN_SIZE,
                self.config.TIME_BIN_SIZE
            )
            # If it so happens that the trial doesn't contain spikes in this buffer, trim the bins (this doesn't happen often when we have large ISIs leading the trial)
            first_spike_time = min(*event["spikes_source_timestamp"])
            if first_spike_time > bins[0]:
                bins = bins[np.digitize(first_spike_time, bins) - 1:]

            # Note that a 1s, 100Hz stim train actually only spans 990ms, and will be binned into 495 bins...

            stim_trial = torch.zeros(len(bins) - 1, valid_stim, 1, dtype=torch.float)
            record_trial = torch.zeros(len(bins) - 1, valid_record, 1, dtype=torch.float)
            # Right now we only encode existence/bin count,
            # - we should definitely add attributes like stim amplitude
            # - and possibly info like timing within bin
            COUNT_FEAT_INDEX = 0

            # * Encode stim
            # This will need revisiting when we have:
            # - multi-channel stimulation
            # - recording more features/non-constant pulses
            channel = event["metadata_channel"]
            if isinstance(channel, np.uint8):
                channel = [channel] # May eventually have multi-channels
            channel_event_times = torch.tensor(event["sync_source_timestamp"]) # TODO multichannel - should filter events by channel
            # In the future, we don't need a for loop.
            # We can directly pass the list of channel indices for the events
            for c in channel:
                event_bins = np.digitize(channel_event_times, bins, right=False) # 1 - N-1
                # Specifically, we should replace ^ with an actual encoding of the stim (pulled from metadata, if fixed per trial).
                bin_indices = event_bins - 1 # bin_idx = 0 implies event is between bin bound 0 and 1.
                intra_bin_timing = channel_event_times - bins[bin_indices] # Should be, on average, SPIKE_TIMING_OFFSET
                # Odds of timing being exactly zero are ~0.
                valid_bin_filter = (bin_indices >= 0) & (bin_indices < (len(bins) - 1))
                # ! Note that if multiple events are in the same index, we currently just multiply assign (only one will persist)
                # For more general case, seems like we should `scatter` info across another dim and then squash
                # Or numpy indexing may be sufficient... cf below spike scatter
                STIM_PEDESTAL = None # ! not using this info right now
                stim_trial[
                    bin_indices[valid_bin_filter],
                    stim_channel_map[(c, STIM_PEDESTAL)],
                    COUNT_FEAT_INDEX
                ] = intra_bin_timing[valid_bin_filter].float()

            # * Encode record
            spike_events = torch.tensor(event["spikes_source_timestamp"])
            spike_feats = torch.ones_like(spike_events, dtype=torch.float) # Simply counting is enough for now
            spike_bins = np.digitize(spike_events, bins, right=False)
            bin_indices = spike_bins - 1
            valid_bin_filter = (bin_indices >= 0) & (bin_indices < (len(bins) - 1))
            # spikes_channel = event["spikes_channel"] + _PEDESTAL_OFFSET * event["spikes_source_index"]
            spikes_channel_slotted = np.array([record_channel_map[(c,p)] for c, p in zip(
                event["spikes_channel"][valid_bin_filter], event["spikes_source_index"][valid_bin_filter]
            )])
            valid_channel_filter = spikes_channel_slotted != -1
            # ! valid_bin_filter = valid_bin_filter & (spikes_channel_slotted != -1)
            # * We don't & with valid_bin_filter because valid_channel_filter is on bin-filtered events already.. (mismatch in size)
            spikes_channel_slotted = spikes_channel_slotted[valid_channel_filter]

            # Option 1: np assign. Will overwrite multiple events in the same index, but this seems like a safe assumption at 2ms
            # record_trial[
            #     bin_indices[valid_bin_filter],
            #     spikes_channel_slotted,
            #     COUNT_FEAT_INDEX
            # ] = spike_feats[valid_bin_filter]

            # Option 2: vectorized np assign. By guaranteeing events are unique in event dimension, we can then control how to collapse
            # Requires considerable memory
            # Fairly certain torch.gather would work without expanding here...

            # temp_trial = torch.zeros_like(record_trial).expand(-1, -1, len(bin_indices[valid_bin_filter])).clone()
            # temp_trial[
            #     bin_indices[valid_bin_filter][valid_channel_filter],
            #     spikes_channel_slotted,
            #     np.arange(len(bin_indices[valid_bin_filter][valid_channel_filter])) # spread along unique range as overkill method of not clobbering spikes that repeat bin/channel
            # ] = spike_feats[valid_bin_filter][valid_channel_filter]
            # record_trial[..., COUNT_FEAT_INDEX] = temp_trial.sum(-1)

            # Option 3: For loop over spike snippets
            for i, event_bin in enumerate(bin_indices[valid_bin_filter][valid_channel_filter]):
                record_trial[
                    event_bin,
                    spikes_channel_slotted[i],
                    COUNT_FEAT_INDEX
                ] += spike_feats[valid_bin_filter][valid_channel_filter][i]

            stim_states.append(stim_trial)
            record_states.append(record_trial)
        return stim_states, record_states

    # ===
    #  Utilities and time cropping
    # TODO: convenience fns for extracting different epochs, pre/post/during stim
    # ===
    def bind_session_remapper(self, target_session_order: Tuple[SessionSet, ...], self_session_alias: Optional[Union[SessionSet, Tuple[SessionSet, ...]]]=None):
        r"""
            Dataset serves sessions as idxes for model to embed using its own config.
            We may in general be providing data to a different model.
            Use this remap to provide the model with what it expects.
            `self_session_alias`: Override self.session if you believe self.session can be related to a different session that _is in_ target session order. e.g. same session, diff set.
        """

        session_idx_map = []
        session_src = self.session if self_session_alias is None else self_session_alias
        if self.multisession:
            for session in session_src:
                if session in target_session_order:
                    session_idx_map.append(target_session_order.index(session))
                else:
                    session_idx_map.append(-1)
                    print(f"Warning: session {session} found in dataset that is novel to targeted model, will map to -1 and likely error.")
        else:
            session_idx_map = [target_session_order.index(session_src)]
        self.session_idx_map = torch.tensor(session_idx_map)

    @staticmethod
    def extract_window(all_timeseries, window_left, window_right, centers=None):
        r"""
            all_timeseries: tuple of series to extract from, each length B
            centers: length B, alignment point if provided, window_left and right are treated as relative distances instead of absolute edges.
            # * Will stack outputs if centers if provided.
        """
        def extract(series):
            extracted = []
            for i, el in enumerate(series):
                left = window_left[i] if isinstance(window_left, torch.Tensor) else window_left
                right = window_right[i] if isinstance(window_right, torch.Tensor) else window_right
                if centers is not None:
                    left = centers[i] - left
                    right = centers[i] + right
                extracted.append(el[int(left):int(right)])
            return torch.stack(extracted) if centers is not None else extracted
        return [extract(series) for series in all_timeseries]

    def crop_to_stim_windows(self, pulse_start=0, pulse_end=-1):
        r"""
            Stim epoch cropper. Intended to be used before squaring/alignment, but should work after as well.
            pulse_start: reference left pulse. Intended to be set to 0 or higher. Will not crop if set to None
            pulse_end: reference right pulse. Same syntax.
        """
        if pulse_start is None:
            stim_left = torch.zeros(len(self.stim_states))
        else:
            stim_left = torch.tensor([stim[..., 0].sum(1).nonzero()[pulse_start, 0] for stim in self.stim_states])
        if pulse_end is None:
            stim_right = torch.tensor([len(stim) for stim in self.stim_states])
        else:
            stim_right = torch.tensor([stim[..., 0].sum(1).nonzero()[pulse_end, 0] for stim in self.stim_states])
        self.record_states, self.stim_states = ICMSDataset.extract_window(
            (self.record_states, self.stim_states), stim_left, stim_right
        )

    def align_and_min_crop_trials(self, align_stim_index=0, align_index:Union[None, np.ndarray]=None):
        r"""
            Align states and crop minimally (just enough to square off arrays).
            If align_stim_index is provided, assumes there are stim in current timeframes, and aligns to stim pulse.
            Else, uses hardcoded index across arrays as alignment point.
        """
        if self.aligned:
            return
        assert align_stim_index is not None or align_index is not None, "must provide alignment point"
        if align_stim_index is not None:
            align_times = np.array([stim[..., align_stim_index].sum(1).nonzero()[0, 0] for stim in self.stim_states])
        else:
            align_times = np.array(align_index)
        left = align_times.min()
        right = np.array([r.size(0) - at for r, at in zip(self.record_states, align_times)]).min()
        self.record_states, self.stim_states = ICMSDataset.extract_window(
            (self.record_states, self.stim_states),
            left, right, align_times
        )
        self.aligned = True

    def crop_pre_stim(self):
        self.crop_to_stim_windows(None, 0)
        self.align_and_min_crop_trials(align_stim_index=None, align_index=[len(s)-1 for s in self.stim_states]) # add a -1 so we don't get weird wrapping.
    def crop_post_stim(self):
        self.crop_to_stim_windows(-1, None)
        self.align_and_min_crop_trials(align_stim_index=None, align_index=np.ones(len(self.stim_states))) # similarly, add a +1
    def crop_in_stim(self):
        self.crop_to_stim_windows(0, -1)
        self.align_and_min_crop_trials(align_stim_index=1)
    # Note these cropping methods are more or less incompatible with PSTH

    def trim_trials(self,
        stim_states: List[torch.Tensor],
        record_states: List[torch.Tensor],
        should_align_to_first_stim=True
    ):
        r"""
            stim_states: Lism[T x C x N]
            record_states: List[T x C x N] (same length)
            align_to_first_stim:
                If true, identifies first stim and takes prefix/suffix based on config.
                If false, takes a fixed chop of trials as given
            # ! Note that we sometimes lose the last pulse, due to offsets... cest la vie.
        """
        def _align_to_first_stim(record, stim):
            first_stim_idx = stim[..., 0].sum(1).nonzero()[0, 0]
            start_idx = first_stim_idx - round(self.config.PREFIX_TIME / self.config.TIME_BIN_SIZE)
            assert start_idx >= 0, f"Insufficient data for prefix {self.config.PREFIX_TIME}"
            return record[start_idx:], stim[start_idx:]
        if should_align_to_first_stim: #
            align_times = np.array(stim[..., 0].sum(1).nonzero()[0, 0] for stim in stim_states)
            record_states, stim_states = zip(*[_align_to_first_stim(r, s) for r, s in zip(record_states, stim_states)])
            self.aligned = True

        # trim the T to a set max length
        lengths = [x.size(0) for x in self.record_states]
        min_length, max_length = min(lengths), max(lengths)
        if max_length - min_length > 300:
            print(f"Note! Trimming dataset with large difference in trial lengths. Max: {max_length}, min: {min_length}")
        trim_length = min(*lengths, math.floor(MAX_DURATION / self.config.TIME_BIN_SIZE))
        # we trim to the minimal available time,
        # as we don't want the model to accidentally learn null activity just because the recording was stopped early
        stim_states = [stim_state[:trim_length] for stim_state in stim_states]
        record_states = [record_state[:trim_length].int() for record_state in record_states] # we just don't need full floats right now
        self.trimmed = True
        return torch.stack(stim_states, 0).float(), torch.stack(record_states, 0)

    def __len__(self):
        return len(self.record_states)

    def __getitem__(self, index):
        # return recordings, stim for a trial, and some optional metadata
        session = self.key_df.iloc[index]['session_idx'] if self.multisession else torch.tensor(DEFAULT_SESSION_IDX)
        if self.session_idx_map:
            session = self.session_idx_map[session]
        return (
            self.record_states[index],
            self.stim_states[index],
            session,
            None if self.teacher_states is None else self.teacher_states[index],
        )

class Kernel(Enum):
    GAUSSIAN = 'gaussian'
    HALF_GAUSSIAN = 'half_gaussian'

class DataManipulator:
    r"""
        Utility class. Refactored out of `ICMSDataset` to keep that focused to dataloading.
    """

    @staticmethod
    def kernel_smooth(
        spikes: torch.Tensor,
        window
    ) -> torch.Tensor:
        window = torch.tensor(window).float()
        window /=  window.sum()
        # Record B T C
        b, t, c = spikes.size()
        spikes = spikes.permute(0, 2, 1).reshape(b*c, 1, t).float()
        # Convolve window (B 1 T) with record as convolution will sum across channels.
        window = window.unsqueeze(0).unsqueeze(0)
        smooth_spikes = torch.nn.functional.conv1d(spikes, window, padding="same")
        return smooth_spikes.reshape(b, c, t).permute(0, 2, 1)

    @staticmethod
    def gauss_smooth(
        spikes: torch.Tensor,
        bin_size: float,
        kernel_sd=0.05,
        window_deviations=7, # ! Changed bandwidth from 6 to 7 so there is a peak
        past_only=False
    ) -> torch.Tensor:
        r"""
            Compute Gauss window and std with respect to bins

            kernel_sd: in seconds
            bin_size: in seconds
            past_only: Causal smoothing, only wrt past bins - we don't expect smooth firing in stim datasets as responses are highly driven by stim.
        """
        # input b t c
        gauss_bin_std = kernel_sd / bin_size
        # the window extends 3 x std in either direction
        win_len = int(window_deviations * gauss_bin_std)
        # Create Gaussian kernel
        window = signal.gaussian(win_len, gauss_bin_std, sym=True)
        if past_only:
            window[len(window) // 2 + 1:] = 0 # Always include current timestep
            # if len(window) % 2:
            # else:
                # window[len(window) // 2 + 1:] = 0
        return DataManipulator.kernel_smooth(spikes, window)


    @staticmethod
    def empirical_rates(
        spikes: torch.Tensor,
        bin_size: float,
        kernel_sd=0.05,
        normalize_hz=True
    ) -> torch.Tensor:
        smth_spikes = DataManipulator.gauss_smooth(
            spikes, bin_size=bin_size, kernel_sd=kernel_sd, past_only=True
        )
        if normalize_hz:
            smth_spikes = smth_spikes / bin_size
        return smth_spikes

    @staticmethod
    def empirical_rates_from_dataset(
        dataset: ICMSDataset,
        **kwargs
    ) -> torch.Tensor:
        return DataManipulator.empirical_rates(
            dataset.record_events,
            dataset.bin_size,
            **kwargs
        )

DEFAULT_SWEEP: List[Tuple] = [
    # stim driven
    *[(Kernel.HALF_GAUSSIAN, k) for k in [0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]],
    # broader
    *[(Kernel.GAUSSIAN, k) for k in np.arange(0.01, 0.05, 0.01)],
    *[(Kernel.GAUSSIAN, k) for k in np.arange(0.06, 0.2, 0.02)],
]

class PSTHDataset:
    r"""
        Mashup between PSTH class responsibilities vs PSTH utilities.
        PSTH requires alignment of data across trials.
        For clarity in notebooks, when retrieving PSTH from class, this alignment must be done explicitly on underlying dataset first.
        # ! PSTH utilities assume alignment of input data.
        # ! Supports psth-ing of conditions subsets, but caching uniqueness is just based on length of dataset, beware of collisions.
        Convention: `get` prefix indicates class methods, `make` indicates utility.
        TODO:
        - kernel cv sweep
        - make compatible with dataset subset trials
    """

    def __init__(
        self, dataset: ICMSDataset,
        cache_dir='/home/joelye/projects/icms_modeling/data/psth/',
        kernel_sd=0.005, # default, overridden if sweeping
        kernel=Kernel.GAUSSIAN # default, overridden if sweeping.
    ):
        self.dataset = dataset
        if not dataset.pristine:
            print("Warn: Computing PSTH on non-pristine dataset. This may lead to unexpected results.")
        self.cache_dir= cache_dir
        self.kernel_sd = kernel_sd
        self.kernel = kernel
        self.conditions = None
        self.psth = None
        self.loss = nn.PoissonNLLLoss(log_input=True)

    def get_conditions(self):
        return self.dataset.get_conditions()

    @staticmethod
    def id_conditions_for_survey(stim: torch.Tensor):
        r"""
        Only for use in survey data (legacy)
        # stim, B T C 1 -> B
        # Stim condition should be metadata, but currently, we manually infer it
        # As the 1 channel that is active during the trial
        # Returns dict of channel idx to trial idxes for that condition
        """
        active_channel = stim.sum(1)[..., 0] # B x C
        _, active_channel = active_channel.nonzero(as_tuple=True) # B
        condition_idxes = defaultdict(list)
        for trial_idx, channel in enumerate(active_channel):
            condition_idxes[channel.item()].append(trial_idx)
        return condition_idxes

    def get(
        self,
        as_trials=False,
        normalize_hz=False,
        should_sweep=True,
        sweep_per_channel=True,
        sweep_per_condition=True,
        sweep_params: List[Tuple]=DEFAULT_SWEEP,
        reload=False,
    ) -> Union[torch.Tensor, ConditionDict]:
        assert self.dataset.aligned
        r"""
            as_trials:
                If true, repeat PSTH along trials to be len(dataset) (torch.Tensor return)
                Else, returns PSTH according to conditions, (as ConditionDict)
            normalize_hz:
                Return rates in Hz if true, else in spikes/bin
        """
        def normalize(val):
            if isinstance(val, dict):
                return {k: normalize(v) for k, v in val.items()}
            return val / self.dataset.config.TIME_BIN_SIZE if normalize_hz else val
        psth = self._get_psth_as_conditions(
            should_sweep=should_sweep,
            sweep_per_channel=sweep_per_channel,
            sweep_per_condition=sweep_per_condition,
            sweep_params=sweep_params,
            reload=reload,
        )
        if not as_trials:
            return normalize(psth)
        return normalize(
            PSTHDataset.conditions_as_trials_for_idxes(psth, self.get_conditions())
        )

    def trial_rates_as_condition_averages(
        self,
        trial_rates: torch.Tensor,
        conditions: Optional[ConditionInfo] = None
    ) -> torch.Tensor:
        r"""
            Converts trial rates (computed from `self.dataset`) to condition averaged trial rates.
            If conditions is provided, only outputs for those conditions/trials (defaults to own full conditions)
        """
        return PSTHDataset.conditions_as_trials_for_idxes(
            self.get_from_rates(trial_rates),
            conditions if conditions else self.get_conditions()
        )

    @staticmethod
    def conditions_as_trials_for_idxes(psth: ConditionDict, condition_idxes: ConditionInfo):
        r"""
            Places psth tensors into trials given in condition_idxes
            Assumes trials form a contiguous range (as items are stacked into tensor)
            Assumes that the condition keys referred to in condition_idxes are the same as those in psth.

            Old docs:
                # Given PSTH of size Conds x Time x Channels, condition_idxes dict, with idxes matching PSTH idxes
                # FYI: This makes big assumptions about the ordering of PSTH tensor
                # Specifically, assumes that PSTHs dim 0 is ordered as condition indices. This should be the case in any normal generation procedure, since the PSTH is calculated by a call to the same underlying df (see end of `make_psth`).
                # However, if we're predicting trials for subset of conditions, psth may be "full" i.e. from the original dataset or calculated from the subset dataset.
                # ! If the latter, set `is_subset_psth` to True. (This path is ! not well tested)
                # ! `is_subset_psth` assumes ordering of condition_idxes matches ordering of PSTH tensor (this is typically true, but better to not rely on this unless declared)
                # Normally we expect to use full dataset PSTHs as a reference, though, so in this case we will index the PSTH tensor with condition index.
        """
        trials_queried = np.sort(np.concatenate([np.array(v) for v in condition_idxes.values()])) # assumes zero indexed
        assert (trials_queried == np.arange(len(trials_queried))).all()
        _sample_psth = next(iter(psth.values()))
        psth_as_trials = torch.zeros(len(trials_queried), *_sample_psth.size())
        for c in condition_idxes:
            psth_as_trials[condition_idxes[c]] = psth[c]
        return psth_as_trials

    def evaluate_loss(self, rates: torch.Tensor, spikes: torch.Tensor):
        rates[rates <= 0] = 1e-8
        return self.loss(rates.log(), spikes)

    @staticmethod
    def _subset_conditions(conditions, keys):
        # be careful about using any subsetted condition output for assignment as predictions
        return {
            cond: [ci for ci in conditions[cond] if ci in keys] for cond in conditions
        }

    def train_val_nll(self, records, kernel, kernel_sd, conditions, train_ratio=0.8, seed=42):
        # records: B x T, or B x T x C
        # TODO implement cross-validation, not simple validation as done here
        keys = np.arange(records.size(0))
        np.random.seed(seed)
        np.random.shuffle(keys)
        tv_cut = int(train_ratio * len(keys))
        if records.ndim == 2:
            records = records.unsqueeze(-1)
        train_key, val_key = keys[:tv_cut], keys[tv_cut:]
        train_conditions = PSTHDataset._subset_conditions(conditions, train_key)
        condition_psth = PSTHDataset.make_psth(
            records, # because it's difficult to appropriately update conditions to new idxes in subset, simply don't subset and let PSTH function extract appropriate idxes
            train_conditions,
            bin_size=self.dataset.bin_size,
            kernel_sd=kernel_sd,
            kernel=kernel
        )
        psth_as_trials = PSTHDataset.conditions_as_trials_for_idxes(condition_psth, conditions)
        val_as_trials = psth_as_trials[val_key] # Do this because it's unclear we can get val_as_trials directly easily

        # train_psth = PSTHDataset.conditions_as_trials_for_idxes(condition_psth, train_conditions)
        # print(f"tv loss. Kernel: {kernel, kernel_sd} Train: {self.evaluate_loss(train_psth, records[train_key]), self.evaluate_loss(psth_as_trials, val_records)}")
        return self.evaluate_loss(val_as_trials, records[val_key])

    def _get_swept_psth(
        self,
        sweep_per_channel=True,
        sweep_per_condition=True,
        sweep_params=DEFAULT_SWEEP
    ):
        # ! there's a better implementation that reformats channels into conditions.
        records = self.dataset.record_events # BTC
        def get_single_swept_psth(
            record: torch.Tensor,
            conditions: ConditionInfo
        ):
            # record: # B x T (x C)
            # conditions: conds to group, values should idx all of B.
            # return: # 1 x T
            squeezed = record.ndim == 2
            sweep_scores = np.array([
                self.train_val_nll(
                    record, kernel=p[0], kernel_sd=p[1], conditions=conditions
                ) for p in sweep_params
            ]) # Min NLL is best
            params = sweep_params[sweep_scores.argmin()]
            if squeezed:
                record = record.unsqueeze(-1)
            _psth = PSTHDataset.make_psth(
                record,
                conditions,
                bin_size=self.dataset.bin_size,
                kernel=params[0],
                kernel_sd=params[1]
            )
            if squeezed:
                _psth = {k: v.squeeze(-1) for k, v in _psth.items()}
            return _psth

        def get_channel_psth(record: torch.Tensor, conditions: ConditionInfo) -> ConditionDict:
            # record: B T C
            # conditions: to sweep together
            # return: 1 T C
            if sweep_per_channel:
                out = {}
                for c in range(record.size(2)):
                    out[c] = get_single_swept_psth(record[..., c], conditions)
                # stack channels
                stack_out = {}
                for cond in conditions: # now TC
                    stack_out[cond] = torch.stack([out[c][cond] for c in range(record.size(2))], dim=-1)
                return stack_out
            else:
                return get_single_swept_psth(record, conditions)

        conditions = self.get_conditions()
        psth = {}
        if sweep_per_condition:
            # Parallel
            # mp_units = len(os.sched_getaffinity(0))
            # def _cond_psth(cond_key):
            #     cond_idx = conditions[cond_key]
            #     cond_record = records[cond_idx] # B x T x C
            #     _sub_cond = {cond_key: np.arange(cond_record.size(0))}
            #     return get_channel_psth(cond_record, _sub_cond)
            # with Pool(mp_units) as p:
            #     psth = list(tqdm(p.imap(
            #         _cond_psth,
            #         conditions.keys()
            #     ), total=len(conditions)))
            # Serial
            for cond_key, cond_idx in tqdm(conditions.items()):
                cond_record = records[cond_idx] # B x T x C
                _sub_cond = {0: np.arange(cond_record.size(0))} # Always condition 0 as far as inner functions are concerned
                _sub_psth = get_channel_psth(cond_record, _sub_cond)[0]
                psth[cond_key] = _sub_psth
        else:
            psth = get_channel_psth(records, conditions)
        # Cond x T x C
        return psth

    def _get_psth_as_conditions(
        self,
        full_res=False,
        should_sweep=True,
        sweep_per_channel=True,
        sweep_per_condition=True,
        sweep_params=DEFAULT_SWEEP,
        reload=False,
        **kwargs
    ) -> ConditionDict:
        if self.psth is None or reload:
            if should_sweep:
                cache_path = Path(self.cache_dir) / f"{self.dataset.cache_hash}_sweep.psth"
            else:
                cache_path = Path(self.cache_dir) / f"{self.dataset.cache_hash}_{self.kernel}_{self.kernel_sd}.psth"
            if not reload and self.dataset.pristine and cache_path.exists(): # Attempt to retrieve cache
                self.psth = torch.load(cache_path)
            if self.psth is None:
                if should_sweep:
                    self.psth = self._get_swept_psth(
                        sweep_per_channel=sweep_per_channel,
                        sweep_per_condition=sweep_per_condition,
                        sweep_params=sweep_params
                    )
                else:
                    assert not full_res, "Full res not implemented."
                    bin_size = self.dataset.config.TIME_BIN_SIZE
                    conditions = self.get_conditions()
                    self.psth = PSTHDataset.make_psth(
                        self.dataset.record_events,
                        conditions,
                        kernel_sd=self.kernel_sd,
                        kernel=self.kernel,
                        bin_size=bin_size,
                    )
                if self.dataset.pristine:
                    torch.save(self.psth, cache_path)
                else:
                    print("Warn: Computed PSTH on non-pristine dataset")
        return self.psth

    @staticmethod
    def make_psth(
        raw_data: torch.Tensor,
        condition_idxes: Dict[int, List[int]],
        bin_size=0.001,
        kernel_sd=0.05,
        kernel=Kernel.GAUSSIAN
    ) -> ConditionDict:
        """
            args:
                raw_data: B x T x C, full resolution, aligned as desired
                condition_idxes: dict of lists of indices (into trial B), per condition
                bin_size: Target bin width. Note, this doesn't do resampling, bin size is for giving scale to kernel_sd
            return:
                psths: [Cond x TxC], in the same order as condition_idxes (of len(condition_idxes))
        """
        if kernel == Kernel.GAUSSIAN:
            smooth_data = DataManipulator.gauss_smooth(raw_data, bin_size=bin_size, kernel_sd=kernel_sd, past_only=False)
        elif kernel == Kernel.HALF_GAUSSIAN:
            smooth_data = DataManipulator.gauss_smooth(raw_data, bin_size=bin_size, kernel_sd=kernel_sd, past_only=True)
        else:
            assert False
        return {cond: smooth_data[idxes].mean(0) for cond, idxes in condition_idxes.items()}

    @staticmethod
    def make_psth_from_rates(
        rates: torch.Tensor,
        dataset_conditions: ConditionInfo
    ) -> ConditionDict:
        return {cond: rates[cond_idxes].mean(0) for cond, cond_idxes in dataset_conditions.items()}

    def get_from_rates(
        self,
        rates: torch.Tensor
    ):
        return PSTHDataset.make_psth_from_rates(rates, self.get_conditions())

@attr.define
class DatasetAttrs:
    stim_channels: np.ndarray # unique channels. Not necessarily just stim-ed channels. 1-indexed
    record_channels: np.ndarray # can be N. 1-indexed
    record_mean_std: torch.Tensor
    stim_channel_dim: int
    subject: str
    # Metadata stating which electrode array was used (we always stim both sensory arrays)
    # ! This is currently consumed by `ICMSDataset`, likely not actually needed in interface (model only needs to know `channels``), but we pass it around just in case atm
    record_array: str
    bin_size: float
    teacher_mean_std: Optional[torch.Tensor]
    # Some channel props are mutable/time-varying (e.g. amp recorded)
    # Some are constant (space)
    # Hypothetically some things could be trial-varying (but let's not worry about that for now)
    # 1. how do I deal?
    #   This is easy, when assembling, dataset will provide time-varying PER channel
    #   The output vector pre-assigns channel IDs across the channel dimension
    #   e.g. T x C x N (=1 for amp)
    #   Embedding module has channel IDs converted to embeddings on hand
    #   e.g. C x M
    #   We do NOT want to just concat, or even dim-expand and add; because the RNN collapses channel dimension (only takes along time)
    #       - Also note that it's pointless to only embed an ID (bijects with channel dimension)
    #   e.g. TODO unclear how to "infuse" the channel embedding into a specific time-varying, since things will shortly be collapsed
    #   (not a problem in transformer land if we also split by channels, but then we'll have e.g. 100 x Time mini-elements getting transformed...)
    #   * One proposal
    #   We can quickly MLP transform time-varying + non-time-varying bits together into a new channel rep
        # Also cr Vahagn; if we're only embedding spatial info (not i.e. impedance,) we may consider conv-ing somehow.
    # 2. what about channels that stim only, record only, or both?
    #   I will likely just provide these pieces of info separately
    #   The embedder will provide the same position information if necessary
    # Let's denote embedding-enhanced dimensions with ' e.g. N -> N'
    # When channels are provided separately as RNN input
    # C_stim x N' cat C_rec x M' -- it's all flattened, but all the info is there...
    # Alternatively, combined should be represented as
    # C_comb x (N+M)'
    # MLP applies twice in former and once in latter. nbd?

def load_distilled_rates(path, dataset: ICMSDataset):
    distill_pth = Path(path).expanduser()
    distill_rates_tv = torch.load(distill_pth, map_location='cpu')
    distill_rates = torch.zeros((
        distill_rates_tv['train_rates'].size(0) + distill_rates_tv['val_rates'].size(0),
        *distill_rates_tv['train_rates'].size()[1:]
    ), dtype=torch.float)

    pl.seed_everything(DATASET_SHUFFLE_KEY)
    keys = dataset.unique_keys
    np.random.shuffle(keys)
    tv_cut = int(0.8 * len(keys))
    train_keys, val_keys = keys[:tv_cut], keys[tv_cut:]
    train_idxes, val_idxes = dataset.get_key_indices(train_keys), dataset.get_key_indices(val_keys)
    distill_rates[train_idxes] = distill_rates_tv['train_rates']
    distill_rates[val_idxes] = distill_rates_tv['val_rates']

    # Quick validation chec
    check_spikes = torch.zeros_like(distill_rates)
    check_spikes[train_idxes] = distill_rates_tv['train_spikes'].float()
    check_spikes[val_idxes] = distill_rates_tv['val_spikes'].float()
    assert (check_spikes == dataset.record_states[...,0]).all()
    return distill_rates.log()

# * PTL natively supports dict items and so this collater can definitely be cleaned up.
def collater_factory(pad: bool = False):
    def inner(batch) -> ICMSBatch:
        def group_item(batch_entries: Union[Iterable[torch.tensor], Iterable[int], Iterable[None]]):
            if batch_entries[0] is None:
                return None
            if not torch.is_tensor(batch_entries[0]):
                batch_entries = [torch.tensor(entry) for entry in batch_entries]
            if pad and batch_entries[0].ndim > 0:
                # We use batch first because entire codebase is structured that way
                return pad_sequence(batch_entries, batch_first=True)
            return torch.stack(batch_entries)
        items = [group_item(item_batch) for item_batch in zip(*batch)]
        return ICMSBatch(
            record=items[0],
            stim=items[1],
            session=items[2],
            teacher=items[3],
            lengths=torch.tensor([len(i[0]) for i in batch]) if pad else None
        )
    return inner

def get_dataloader(dataset: ICMSDataset, batch_size=100, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(
        dataset,
        collate_fn=collater_factory(pad=not dataset.trimmed),
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
