import abc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, NamedTuple
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
import logging

from context_general_bci.config import DatasetConfig
from context_general_bci.subjects import SubjectArrayRegistry, SubjectInfo, SubjectName
from context_general_bci.subjects.pitt_chicago import P2, P3
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskRegistry, churchland_misc

# FYI: Inherited dataclasses don't call parent's __init__ by default. This is a known issue/feature:
# https://bugs.python.org/issue43835
StimCommand = NamedTuple("StimCommand", times=np.ndarray, channels=np.ndarray, current=np.ndarray)
CommandPayload = Dict[Path, StimCommand]

logger = logging.getLogger(__name__)

# Onnx requires 3.9, kw_only was added in 3.10. We patch with this suggestion https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses/53085935#53085935
@dataclass
class _ContextInfoBase:
    subject: SubjectInfo # note this is an object/value
    task: ExperimentalTask # while this is an enum/key, currently

@dataclass
class _ContextInfoDefaultsBase:
    _arrays: List[str] = field(default_factory=lambda: []) # arrays (without subject handles) that were active in this context. Defaults to all known arrays for subject
    datapath: Path = Path("fake_path") # path to raws - to be provided by subclass (not defaulting to None for typing)
    alias: str = ""


# Regress for py 3.9 compat
# @dataclass(kw_only=True)
@dataclass
class ContextInfo(_ContextInfoDefaultsBase, _ContextInfoBase):
    r"""
        Base (abstract) class for static info for a given dataset.
        Subclasses should specify identity and datapath
    """
    # Context items - this info should be provided in all datasets.

    # These should be provided as denoted in SubjectArrayRegistry WITHOUT subject specific handles.
    # Dropping subject handles is intended to be a convenience since these contexts are essentially metadata management. TODO add some safety in case we decide to start adding handles explicitly as well...

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
            logging.warning(f"ContextInfo datapath not found ({self.datapath})")

    @property
    def array(self) -> List[str]:
        r"""
            We wrap the regular array ID with the subject so we don't confuse arrays across subjects.
            These IDs will be used to query for array geometry later. `array_registry` should symmetrically register these IDs.
        """
        return [self.subject.wrap_array(a) for a in self._arrays]

    @staticmethod
    def get_id(subject: SubjectName, task: ExperimentalTask, id_str: str):
        return f"{task}-{subject.value}-{id_str}"

    @property
    def id(self):
        return ContextInfo.get_id(self.subject.name, self.task, self._id())

    @abc.abstractmethod
    def _id(self):
        raise NotImplementedError

    @property
    def session_embed_id(self):
        return self.id

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
        logger.info(f"Preprocessing {self.task}: {self.datapath}...")
        return loader.load(
            self.datapath,
            cfg=cfg,
            cache_root=cache_root,
            subject=self.subject,
            context_arrays=self.array,
            dataset_alias=self.alias,
            task=self.task
        )

    # For sorting
    def __eq__(self, other):
        if not isinstance(other, ContextInfo):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, ContextInfo):
            return NotImplemented
        return self.id < other.id

    def __gt__(self, other):
        if not isinstance(other, ContextInfo):
            return NotImplemented
        return self.id > other.id


@dataclass
class _PassiveICMSContextInfoBase:
    session: int
    set: int

    train_dir: Path
    stim_banks: Tuple[int]
    stimsync_banks: Tuple[int]


@dataclass
class PassiveICMSContextInfo(ContextInfo, _PassiveICMSContextInfoBase):

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
        data_root=Path("/home/joelye/projects/icms_modeling/data/binned_pth"),
        stim_train_dir_root=Path("/home/joelye/projects/icms_modeling/data/stim_trains/"),
    ):
        if not data_root.exists():
            logger.warning(f"ICMS root not found, skipping ({data_root})")
            return None

        if stim_banks is None:
            stim_banks = [cls._bank_to_array_name[sb] for sb in cls.get_stim_banks(stim_train_dir_root / train_dir)]
        if stimsync_banks is None:
            stimsync_banks = ["medial_s1", "lateral_s1"]
            # stimsync_banks = stim_banks
        info = PassiveICMSContextInfo(
            subject=P2 if session > 500 else P3,
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
class _ReachingContextInfoBase:
    session: int

@dataclass
class ReachingContextInfo(ContextInfo, _ReachingContextInfoBase):

    def _id(self):
        return f"{self.session}-{self.alias}" # All reaching data get alias

    @classmethod
    def build(cls, datapath_str: str, task: ExperimentalTask, alias: str="", arrays=["main"]):
        datapath = Path(datapath_str)
        if not datapath.exists():
            logger.warning(f"Datapath not found, skipping ({datapath})")
            return None
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
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder not found, skipping ({datapath_folder})")
            return []
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

@dataclass
class GDrivePathContextInfo(ContextInfo):
    # for churchland_misc
    def _id(self):
        return f"{self.datapath}"

    @classmethod
    def build_from_dir(cls, datapath_folder_str: str):
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder not found, skipping ({datapath_folder})")
            return []
        all_info = []
        for path in datapath_folder.glob("*.mat"):
            subject = path.stem.split('-')[0]
            if subject == 'nitschke':
                arrays = ['PMd', 'M1']
            elif subject == 'jenkins':
                arrays = ['PMd', 'M1']
            elif subject == 'reggie':
                arrays = ['PMd', 'M1']
            # find pre-registered path
            all_info.append(GDrivePathContextInfo(
                subject=SubjectArrayRegistry.query_by_subject(subject),
                task=ExperimentalTask.churchland_misc,
                _arrays=arrays,
                datapath=path,
                alias=f'churchland_misc_{path.stem}',
            ))
        return all_info


DYER_CO_FILENAMES = {
    ('mihi', 1): 'full-mihi-03032014',
    ('mihi', 2): 'full-mihi-03062014',
    ('chewie', 1): 'full-chewie-10032013',
    ('chewie', 2): 'full-chewie-12192013',
}
@dataclass
class DyerCOContextInfo(ReachingContextInfo):
    @classmethod
    def build(cls, handle, task: ExperimentalTask, alias: str="", arrays=["main"], root='./data/dyer_co/'):
        datapath = Path(root) / f'{DYER_CO_FILENAMES[handle]}.mat'
        if not datapath.exists():
            logger.warning(f"Datapath not found, skipping ({datapath})")
            return None
        subject = SubjectArrayRegistry.query_by_subject(
            datapath.name.split('-')[-2].lower()
        )
        session = int(datapath.stem.split('-')[-1])
        return DyerCOContextInfo(
            subject=subject,
            task=task,
            _arrays=arrays,
            alias=alias,
            session=session,
            datapath=datapath,
        )

# Data has been replaced with M1 only data
# GALLEGO_ARRAY_MAP = {
#     'Lando': ['LeftS1Area2'],
#     'Hans': ['LeftS1Area2'],
#     'Chewie': ['M1', 'PMd'], # left hemisphere M1
#     'Mihi': ['M1', 'PMd'],
# }

# CHEWIE_ONLY_M1 = [ # right hemisphere M1. We don't make a separate distinction
#     'Chewie_CO_20150313',
#     'Chewie_CO_20150630',
#     'Chewie_CO_20150319',
#     'Chewie_CO_20150629',
# ]

@dataclass
class GallegoCOContextInfo(ReachingContextInfo):
    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        def make_info(datapath: Path):
            alias = f'{task.value}_{datapath.stem}'
            if alias.endswith('_M1'):
                alias = alias[:-3]
            subject, _, date, *rest = datapath.stem.split('_') # task is CO always
            subject = subject.lower()
            if subject == "mihili":
                subject = "mihi" # alias
            subject = SubjectArrayRegistry.query_by_subject(subject)
            session = int(date)
            if subject.name == SubjectName.mihi and session in [20140303, 20140306]: # in Dyer release
                return None
            arrays = ['M1']
            # arrays = GALLEGO_ARRAY_MAP.get(subject.name.value)
            # if alias in CHEWIE_ONLY_M1:
                # arrays = ['M1']
            return GallegoCOContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=alias,
                session=int(date),
                datapath=datapath,
            )
        infos = map(make_info, Path(root).glob("*.mat"))
        return filter(lambda x: x is not None, infos)

# read task subtype from `contexts/pitt_type.yaml`
pitt_metadata = {}
# get path relative to this file

# with open(Path(__file__).parent / 'pitt_type.yaml') as f:
#     pitt_task_subtype = yaml.load(f, Loader=yaml.FullLoader)
#     for date in pitt_task_subtype:
#         for session in pitt_task_subtype[date]:
#             session_num = int(list(session.keys())[0])
#             session_type = list(session.values())[0]
#             pitt_metadata[f'P2Home.data.{session_num:05d}'] = session_type

# # frankly JY no longer remembers how this was sourced
# with open(Path(__file__).parent / 'pitt_blacklist.csv') as f:
#     for line in f:
#         non_co_sessions = line.strip().split(',')
#         for session in non_co_sessions:
#             pitt_metadata[session] = 'default'

@dataclass
class BCIContextInfo(ReachingContextInfo):
    session_set: int = 0

    # def session_embed_id(self):
    #     return f"{self.session}" # Many overlapping sessions from the same day, preserve ID.

    @classmethod
    def build_from_dir_varied(cls, root: str, task_map: Dict[str, ExperimentalTask], arrays=["main"]):
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        def make_info(datapath: Path):
            if datapath.is_dir():
                alias = datapath.name
                subject, _, session = alias.split('.')
                session_set = 0
                session_type = pitt_metadata.get(alias, 'default')
            else: # matlab file
                alias = datapath.stem
                subject, _, session, _, session_set, _, *session_type = alias.split('_')
                session_type = '_'.join(session_type)
                blacklist_check_key = f'{subject}_session_{session}_set_{session_set}'
                if blacklist_check_key in pitt_metadata:
                    session_type = pitt_metadata[blacklist_check_key]
            if subject.endswith('Home'):
                subject = subject[:-4]
            elif subject.endswith('Lab'):
                subject = subject[:-3]
            alias = f'{task_map.get(session_type, ExperimentalTask.unstructured).value}_{alias}'
            # print(f"registering {alias} with type {session_type}, {task_map.get(session_type)}")
            return BCIContextInfo(
                subject=SubjectArrayRegistry.query_by_subject(subject),
                task=task_map.get(session_type, ExperimentalTask.unstructured),
                _arrays=[
                    'lateral_s1', 'medial_s1',
                    'lateral_m1', 'medial_m1',
                ],
                alias=alias,
                session=int(session),
                datapath=datapath,
                session_set=session_set
            )
        infos = map(make_info, Path(root).glob("*"))
        return filter(lambda x: x is not None, infos)


    @classmethod
    def build_from_dir(cls, root: str, task_map: Dict[str, ExperimentalTask], arrays=["main"], alias_prefix=''):
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        def make_info(datapath: Path):
            if datapath.is_dir():
                alias = datapath.name
                subject, _, session = alias.split('.')
                session_set = 0
                session_type = pitt_metadata.get(alias, 'default')
            else: # matlab file
                alias = datapath.stem
                pieces = alias.split('_')
                pieces = list(filter(lambda x: x != '', pieces))
                subject, _, session, _, session_set, _, *session_type, control = pieces
                session_type = '_'.join(session_type)
                blacklist_check_key = f'{subject}_session_{session}_set_{session_set}'
                if blacklist_check_key in pitt_metadata:
                    session_type = pitt_metadata[blacklist_check_key]
                    control = 'default'
            if subject.endswith('Home'):
                subject = subject[:-4]
            elif subject.endswith('Lab'):
                subject = subject[:-3]
            subject = subject[:3].upper() + subject[3:]
            alias = f'{alias_prefix}{task_map.get(control, ExperimentalTask.pitt_co).value}_{subject}_{session}_{session_set}_{session_type}'
            if any(i in session_type for i in ['2d_cursor_center', '2d_cursor_pursuit', '2d+click_cursor_pursuit']) or alias_prefix == 'pitt_misc_':
                task = task_map.get(control, task_map.get('default', ExperimentalTask.unstructured))
            else:
                task = task_map.get('default', ExperimentalTask.unstructured)
            return BCIContextInfo(
                subject=SubjectArrayRegistry.query_by_subject(subject),
                task=task,
                _arrays=[
                    'lateral_s1', 'medial_s1',
                    'lateral_m1', 'medial_m1',
                ],
                alias=alias,
                session=int(session),
                datapath=datapath,
                session_set=session_set
            )
        infos = map(make_info, Path(root).glob("*"))
        return filter(lambda x: x is not None, infos)

    @classmethod
    def build_preproc(cls, datapath_folder_str: str, alias_prefix: str = ""):
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder {datapath_folder} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            alias = path.stem
            pieces = alias.split('_')
            subject, _, session, _, session_set, *_ = pieces
            task = ExperimentalTask.pitt_co
            alias = f'{alias_prefix}{task.value}_{subject}_{session}_{session_set}'
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return BCIContextInfo(
                subject=subject,
                task=task,
                _arrays=[
                    'lateral_s1', 'medial_s1',
                    'lateral_m1', 'medial_m1',
                ],
                alias=alias,
                session=int(session),
                session_set=int(session_set),
                datapath=path,
            )
        return map(make_info, datapath_folder.glob("*.pth"))


# Not all have S1 - JY would prefer registry to always be right rather than detecting this post-hoc during loading
# So we do a pre-sweep and log down which sessions have which arrays here
RTT_SESSION_ARRAYS = {
    'indy_20160624_03': ['M1', 'M1_all'],
    'indy_20161007_02': ['M1', 'M1_all'],
    'indy_20160921_01': ['M1', 'M1_all'],
    'indy_20170123_02': ['M1', 'M1_all'],
    'indy_20160627_01': ['M1', 'M1_all'],
    'indy_20160927_06': ['M1', 'M1_all'],
    'indy_20161212_02': ['M1', 'M1_all'],
    'indy_20161011_03': ['M1', 'M1_all'],
    'indy_20161026_03': ['M1', 'M1_all'],
    'indy_20161206_02': ['M1', 'M1_all'],
    'indy_20161013_03': ['M1', 'M1_all'],
    'indy_20170131_02': ['M1', 'M1_all'],
    'indy_20160930_02': ['M1', 'M1_all'],
    'indy_20160930_05': ['M1', 'M1_all'],
    'indy_20161024_03': ['M1', 'M1_all'],
    'indy_20170124_01': ['M1', 'M1_all'],
    'indy_20161017_02': ['M1', 'M1_all'],
    'indy_20161027_03': ['M1', 'M1_all'],
    'indy_20160630_01': ['M1', 'M1_all'],
    'indy_20161025_04': ['M1', 'M1_all'],
    'indy_20161207_02': ['M1', 'M1_all'],
    'indy_20161220_02': ['M1', 'M1_all'],
    'indy_20161006_02': ['M1', 'M1_all'],
    'indy_20160915_01': ['M1', 'M1_all'],
    'indy_20160622_01': ['M1', 'M1_all'],
    'indy_20161005_06': ['M1', 'M1_all'],
    'indy_20161014_04': ['M1', 'M1_all'],
    'indy_20160927_04': ['M1', 'M1_all'],
    'indy_20160916_01': ['M1', 'M1_all'],
    'indy_20170127_03': ['M1', 'M1_all'],
}


@dataclass
class _RTTContextInfoBase:
    date_hash: str

@dataclass
class RTTContextInfo(ContextInfo, _RTTContextInfoBase):
    r"""
        We make this separate from regular ReachingContextInfo as subject hash isn't unique enough.
    """

    def _id(self):
        return f"{self.date_hash}"

    @classmethod
    def build_several(cls, datapath_folder_str: str, arrays=["M1", "M1_all", "S1"], alias_prefix="rtt"):
        r"""
            TODO: not obvious how we can detect whether datapath has S1 or not
        """
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder {datapath_folder} does not exist. Skipping.")
            return []

        def make_info(path: Path):
            subject, date, set = path.stem.split("_")
            subject = SubjectArrayRegistry.query_by_subject(subject)
            date_hash = f"{date}_{set}"
            _arrays = RTT_SESSION_ARRAYS.get(path.stem, arrays)
            return RTTContextInfo(
                subject=subject,
                task=ExperimentalTask.odoherty_rtt,
                _arrays=_arrays,
                alias=f"{alias_prefix}-{subject.name.value}-{date_hash}",
                date_hash=date_hash,
                datapath=path,
            )
        return map(make_info, datapath_folder.glob("*.mat"))

    @classmethod
    def build_preproc(cls, datapath_folder_str: str, alias_prefix: str = "", arrays=["M1"]):
        r"""
            For preprocessed splits produced by dataloaders and `split_eval`. Pytorch already.
        """
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder {datapath_folder} does not exist. Skipping.")
            return []

        def make_info(path: Path):
            subject, date, *tail = path.stem.split("_")
            tail = '_'.join(tail)
            subject = SubjectArrayRegistry.query_by_subject(subject)
            date_hash = f"{date}_{tail}"
            return RTTContextInfo(
                subject=subject,
                task=ExperimentalTask.odoherty_rtt,
                _arrays=arrays,
                alias=f"{alias_prefix}-{subject.name.value}-{date_hash}",
                date_hash=date_hash,
                datapath=path,
            )
        return map(make_info, datapath_folder.glob("*.pth"))



@dataclass
class BatistaContextInfo(ContextInfo):

    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["M1"], alias_prefix="batista", preproc=False):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            if task == ExperimentalTask.cst:
                # Ford_20180627_COCST_TD.mat
                # subject, *_ = root.stem.split("_") # old marino datasets apparently used this?
                subject, *_ = path.stem.split("_")
                if subject == 'Ford':
                    subject = 'batista_f'
                elif subject == 'Earl':
                    subject = 'batista_e'
                arrays = ['main']
            else:
                # data/marino_batista/earl_multi_posture_dco_reaching/DelayedCenterOut_E20210710.mat
                subject = path.parent.name.split('_')[0] # e.g. earl
                arrays = ['M1']
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return BatistaContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{alias_prefix}-{subject.name.value}-{path.stem}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.pth" if preproc else "*.mat"))
        return filter(lambda x: x is not None, infos)

@dataclass
class DeoContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        path = Path(root)
        if not path.exists():
            logger.warning(f"Datapath folder {path} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # should be data
            subject, timestamp = path.name.split('_', 1)
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return DeoContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{task.value}_{subject.name.value}_{path.stem}",
                datapath=path,
            )
        infos = map(make_info, path.glob("*.mat"))
        return filter(lambda x: x is not None, infos)


@dataclass
class FalconContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @staticmethod
    def get_alias(task: ExperimentalTask, subject: SubjectName, stem: str):
        if task == ExperimentalTask.falcon_h1:
            pieces = stem.split('_')
            pre_set_pieces = pieces[:pieces.index('set')]
            stem = '_'.join(pre_set_pieces)
            return f"falcon_{subject.value}-{stem}"
        elif task == ExperimentalTask.falcon_m1:
            if 'behavior+ecephys' in stem:
                session_date = stem.split('_')[-2]
            else:
                session_date = f'ses-{stem.split("_")[1]}'
            return f"falcon_{subject.value}-{session_date}"
        elif task == ExperimentalTask.falcon_h2:
            # sub-T5-held-in-calib_ses-20220518.nwb
            session_date = stem.split('-')[-1]
            return f"falcon_{subject.value}-{session_date}"
        elif task == ExperimentalTask.falcon_m2:
            # sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb
            session_hash = stem.split('ses-')[1].split('_')[0]
            return f"falcon_{subject.value}-{session_hash}"
        return f"falcon_{subject.value}-{stem}"

    @staticmethod
    def get_alias_from_path(task: ExperimentalTask, path: Path):
        subject = path.parts[-3].lower()
        subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
        # Do not differentiate phase split OR set in session for easy transfer - phase split follows set annotation
        if task == ExperimentalTask.falcon_h1:
            pieces = path.stem.split('_')
            pre_set_pieces = pieces[:pieces.index('set')]
            stem = '_'.join(pre_set_pieces)
        elif task == ExperimentalTask.falcon_m1:
            if 'behavior+ecephys' in path.stem:
                session_date = stem.split('_')[-2]
            else:
                session_date = f'ses-{path.stem.split("_")[1]}'
            return f"falcon_{subject.name.value}-{session_date}"
        elif task == ExperimentalTask.falcon_h2:
            # sub-T5-held-in-calib_ses-20220518.nwb
            session_date = stem.split('-')[-1]
            return f"falcon_{subject.value}-{session_date}"
        elif task == ExperimentalTask.falcon_m2:
            # sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb
            session_hash = stem.split('ses-')[1].split('_')[0]
            return f"falcon_{subject.value}-{session_hash}"
        return f"falcon_{subject.name.value}-{stem}"

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["M1"], suffix='', is_dandi=True, alias_prefix=''):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # Note aliases provided here will be explicitly reduced/assigned to sessions in a hardcoded `explicit_session_reduction`
            # Since we need a pretty robust mapper for aliases coming from different places..
            if task == ExperimentalTask.falcon_h1:
                subject = 'h1'
                # subject = path.parts[-3].lower()
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                # Do not differentiate phase split OR set in session for easy transfer - phase split follows set annotation
                # pieces = path.stem.split('_')
                # pre_set_pieces = pieces[:pieces.index('set')]
                # stem = '_'.join(pre_set_pieces)
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{subject.name.value}-{path.stem}",
                    datapath=path,
                )
            elif task == ExperimentalTask.falcon_h2:
                subject = 'h2'
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{subject.name.value}-{path.stem}",
                    datapath=path,
                )
            elif task == ExperimentalTask.falcon_m1:
                # sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb
                # test time: L_20121022_held_out_eval
                subject = "m1"
                # subject = path.split('_')[1]
                # session_date = str(path.stem).split('_')[-2]
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{alias_prefix}{subject.name.value}-{path.stem}",
                    datapath=path,
                )
            elif task == ExperimentalTask.falcon_m2:
                subject = 'm2'
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{alias_prefix}{subject.name.value}-{path.stem}",
                    datapath=path,
                )
        if suffix:
            infos = map(make_info, root.glob(f"*{suffix}*.nwb"))
        else:
            infos = map(make_info, root.glob("*.nwb"))
        return list(filter(lambda x: x is not None, infos))


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