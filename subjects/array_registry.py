from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Type, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# Another registry of sorts - holding subject + array info instead of experimental info.
ArrayID = str
@dataclass
class ArrayInfo:

    @abstractmethod
    def get_channel_count(self) -> int:
        raise NotImplementedError

@dataclass
class GeometricArrayInfo(ArrayInfo):
    r"""
        Contains some metadata.
        There is no way to blacklist indices on the fly currently, e.g. if there's an issue in a particular dataset.
        array: array of electrode indices, possibly reflects the geometry of the implanted array.
    """

    array: np.ndarray
    one_indexed: bool = False

    def as_indices(self):
        r"""
            return indices where this array's data is typically stored in a broader dataset.
        """
        indices = self.array[self.array != np.nan].flatten()
        if self.one_indexed:
            indices = indices - 1
        return indices

    def get_channel_count(self):
        return (self.array != np.nan).sum()

@dataclass
class SortedArrayInfo(ArrayInfo):
    r"""
        Sorted arrays unfortunately have no consistent interface from session to session.
        We support a simple interface; really readin logic (in `model.__init__`) needs to be queried from session info if we want per session layer
    """
    _max_channels: int = 140
    def get_channel_count(self):
        return self._max_channels

@dataclass
class AliasArrayInfo(ArrayInfo):
    # mock some attrs
    arrays: List[ArrayInfo]
    def __init__(self, *arrays: List[ArrayInfo]):
        self.arrays = arrays

    def get_channel_count(self):
        return sum([a.get_channel_count() for a in self.arrays])

class SubjectInfo:
    r"""
        Right now, largely a wrapper for potentially capturing multiple arrays.
        ArrayInfo in turn exists largely to report array shape.
        Provides info relating channel ID (1-indexed, a pedestal/interface concept) to location _within_ array (a brain tissue concept).
        This doesn't directly provide location _within_ system tensor, which might have multi-arrays without additional logic! (see hardcoded constants in this class)
        Agnostic to bank info.
    """
    # 1 indexed channel in pedestal, for each pedestal with recordings.
    # This is mostly consistent across participants, but with much hardcoded logic...
    # By convention, stim channels 1-32 is anterior 65-96, and 33-64 is posterior 65-96.
    # Is spatially arranged to reflect true geometry of array.
    # ! Note - this info becomes desynced once we subset channels. We can probably keep it synced by making this class track state, but that's work for another day...
    _arrays: Dict[ArrayID, ArrayInfo] = {} # Registry of array names. ! Should include subject name as well? For easy lookup?
    _aliases: Dict[ArrayID, List[str]] = {} # Refers to other arrays (potentially multiple) used for grouping

    @property
    def arrays(self) -> Dict[ArrayID, ArrayInfo]:
        return self._arrays

    @property
    def aliases(self) -> Dict[ArrayID, List[ArrayID]]:
        return self._aliases

    def get_channel_count(self, arrays: ArrayID | List[ArrayID] = ""):
        if isinstance(arrays, str) and arrays:
            arrays = [arrays]
        queried = self.arrays.values() if not arrays else [self.arrays[a] for a in arrays]
        return sum([a.get_channel_count() for a in queried])


class SubjectArrayRegistry:
    instance = None
    _subject_registry: Dict[str, SubjectInfo] = {}
    _array_registry: Dict[ArrayID, ArrayInfo] = {}
    _alias_registry: Dict[ArrayID, List[str]] = {}

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    @staticmethod
    def wrap_array(subject_id, array_id):
        return f"{subject_id}-{array_id}"

    # Pattern taken from habitat.core.registry
    @classmethod
    def register(cls, to_register: SubjectInfo, name: Optional[str]=None, assert_type = SubjectInfo):
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls._subject_registry[register_name] = to_register # ? Possibly should refer to singleton instance explicitly
            for array in to_register.arrays:
                cls._array_registry[SubjectArrayRegistry.wrap_array(register_name, array)] = to_register.arrays[array]
            for alias in to_register.aliases:
                cls._array_registry[SubjectArrayRegistry.wrap_array(register_name, alias)] = to_register.aliases[alias]
            return to_register
        return wrap(to_register)

    @classmethod
    def resolve_alias(cls, alias: ArrayID) -> List[ArrayID]:
        return cls._alias_registry[alias]

    @classmethod
    def query_by_array(cls, id: ArrayID) -> ArrayInfo:
        if id in cls._array_registry:
            return cls._array_registry[id]
        elif id in cls._alias_registry:
            return AliasArrayInfo(cls._array_registry[a] for a in cls._alias_registry[id])

    @classmethod
    def query_by_array_geometric(cls, id: ArrayID) -> GeometricArrayInfo:
        query = cls.query_by_array(id)
        assert isinstance(query, GeometricArrayInfo), f"{id} is not a geometric array"
        return query

    @classmethod
    def query_by_subject(cls, id: str) -> SubjectInfo:
        return cls._subject_registry[id]




# ==== Defunct


def get_channel_pedestal_and_location(
    channel_ids: np.ndarray, # either C or C x 2 (already has pedestal info)
    subject,
    normalize_space=True,
    mode="record",
    array_label=None, # ! Config should ideally specify what array is used for stim and record, but this API is overkill for now
):
    assert mode in ["record", "stim"], f"{mode} location extraction not known"
    array_cls = SubjectArrayRegistry.query_by_subject(subject)

    print(f"Info: Extracting {mode} array locations")
    def extract_pedestal_and_loc_within_array(one_indexed_channel_ids: np.ndarray, group_size: int):
        if one_indexed_channel_ids.ndim == 2:
            return one_indexed_channel_ids[:, 0], one_indexed_channel_ids[:, 1]
        return (
            ((one_indexed_channel_ids - 1) % group_size) + 1, # channels should be one-indexed
            ((one_indexed_channel_ids - 1) // group_size).astype(int), # pedestals needn't/shouldn't be
        )
    if mode == "stim":
        channels, pedestals = extract_pedestal_and_loc_within_array(channel_ids, array_cls.channels_per_stim_bank)
    else:
        channels, pedestals = extract_pedestal_and_loc_within_array(channel_ids, array_cls.channels_per_pedestal)

    # TODO - get the right array
    if mode == "record":
        arrays = [*zip(array_cls.motor_arrays, array_cls.sensory_arrays)]
    elif mode == "stim":
        arrays = [*zip(array_cls.sensory_arrays_as_stim_channels)]

    spatial_locs = []
    # Works fine for stim
    # For record - if channel is in stim array, then check sensory array. Else
    def get_coordinates_within_any_array(id, arrays: Tuple[np.ndarray]):
        # arrays are assumed to have mutually exclusive entries
        for arr in arrays:
            matches = torch.tensor(arr == id).nonzero()
            if len(matches):
                return matches[0]
        raise Exception(f"Channel ID {id} not found in any candidate array.")
    for channel, pedestal in zip(channels, pedestals):
        spatial_locs.append(get_coordinates_within_any_array(channel, arrays[pedestal]))
    spatial_locs = torch.stack(spatial_locs, 0)

    if normalize_space:
        _long_side = max([max(*array.shape) for array in array_cls.all_arrays])
        spatial_locs = (spatial_locs.float() - _long_side / 2.) / _long_side

    return torch.tensor(pedestals, dtype=torch.int), spatial_locs

class DummyEmbedding(nn.Module):
    def __init__(self, channel_ids: torch.Tensor, *args, **kwargs):
        super().__init__()
        self.register_buffer('embed', torch.zeros((len(channel_ids), 0), dtype=torch.float), 0)
        self.n_out = 0

    def forward(self):
        return self.embed

class ChannelEmbedding(nn.Module):
    def __init__(
        self,
        channel_ids: torch.Tensor, # ints
        subject: str,
        mode="record", # TODO what does this do?
        array_label=None # TODO what does this do?. Also, what is this?
    ):
        super().__init__()

        # embeds channels by ID (fxn shouldn't discriminate b/n stim/recording, but do use separate weights for ID-ing respective fxn)
        # * Note, this should be embedded st we can generalize to unseen channels
        # e.g. we shouldn't just throw in the ID
        PEDESTAL_FEATS = 4
        SPATIAL_FEATS = 2 # 2D Location

        pedestals, self.spatial_embeddings = get_channel_pedestal_and_location(
            channel_ids,
            subject,
            normalize_space=True,
            mode=mode,
            array_label=array_label
        )

        # Note, locations within a pedestal are embedding uniformly; network can transform it later if it needs to
        num_pedestals = len(np.unique(pedestals))

        # Option 1 - something about gradients not getting cleared here...
        # pedestal_embedder = nn.Embedding(num_pedestals, PEDESTAL_FEATS)
        # self.pedestal_embeddings = pedestal_embedder(pedestals) # N -> N x PEDESTAL_FEATS
        # Option 1.5 - same issue
        # pedestal_embedder = nn.Parameter(torch.zeros(num_pedestals, PEDESTAL_FEATS))
        # self.pedestal_embeddings = pedestal_embedder[pedestals.long()] # N -> N x PEDESTAL_FEATS

        # Option 2
        self.register_buffer("pedestals", pedestals)
        self.pedestal_embedder = nn.Embedding(num_pedestals, PEDESTAL_FEATS)

        self.n_out = PEDESTAL_FEATS + SPATIAL_FEATS

    # TODO optimize so I don't forward call by manually initing a bunch of pedestal embeddings
    # If nn.embedding is the problem
    def forward(self):
        pedestal_embeddings = self.pedestal_embedder(self.pedestals) # C -> C x PEDESTAL_FEATS
        self.spatial_embeddings = self.spatial_embeddings.to(pedestal_embeddings.device)
        # self.spatial_embeddings = self.spatial_embeddings.to(self.pedestal_embeddings.device)

        return torch.cat([
            # self.pedestal_embeddings, self.spatial_embeddings # C x N
            pedestal_embeddings, self.spatial_embeddings # C x N
        ], -1)


