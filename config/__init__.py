from typing import List, Optional, Union, Any, Tuple
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

LENGTH = 'length'

# Convention note to self - switching to lowercase, which is more readable and much less risky now that
# config is typed
class Architecture(Enum):
    ndt = 'ndt'

class Task(Enum):
    icms_one_step_ahead = 'icms_one_step_ahead'
    infill = 'infill'

    # Time-varying (and/or encoder-decoder)
    kinematic_decoding = 'kinematic_decoding'

    # Trial-summarizing
    detection_decoding = 'detection_decoding'

class Metric(Enum):
    # Monitoring metrics to log. Losses are automatically included in lgos.
    bps = 'bps'
    co_bps = 'co-bps'
    kinematic_r2 = 'kinematic_r2'

class Output(Enum):
    # Various keys for different vectors we produce
    rates = 'rates'

@dataclass
class TaskConfig:
    r"""
        The model will be trained on various tasks.
        For more flexibility, we separate model task requirements from dataset task requirements (see 'keys' args below)
        (but this maybe should be revisited)
    """
    task: Task = Task.icms_one_step_ahead

    # infill
    mask_ratio: float = 0.5
    mask_token_ratio: float = 0.8
    mask_random_ratio: float = 0.1

    metrics: List[Metric] = [Metric.bps]

@dataclass
class TransformerConfig:
    n_state: int = 256
    n_heads: int = 4
    n_layers: int = 4
    feedforward_factor: float = 1.
    dropout: float = 0.2 # applies generically

    # causal: bool = True # Pretty sure this should be passed in by task, not configured

    # Position
    learnable_position: bool = False
    max_trial_length: int = 1500

class EmbedStrat(Enum):
    # Embedding strategies
    none = "" # Just ignore context
    token = 'token' # Embed context as a token
    concat = 'concat' # concat embedding and downproject

    project = 'project' # just for array inputs

@dataclass
class ModelConfig:
    hidden_size: int = 256 # For parts outside of backbones
    arch: str = Architecture.ndt
    transformer: TransformerConfig = TransformerConfig()

    half_precision: bool = True
    lr_init: float = 0.0001
    weight_decay: float = 0.0

    # The objective. Not intended to be multitask right now; intent is pretrain/fine-tune.
    task: TaskConfig = TaskConfig()

    # Spike prediction tasks
    lograte: bool = True

    # A few possible strategies for incorporating context information
    # "token" (this is the simplest and thus ideal one)
    # "add" (add representations)
    # "project" (have a context-specific read-in layer)
    # "" - ignore context

    # Trial level
    session_embed_strategy: str = EmbedStrat.token
    session_embed_size: int = 256 # TODO can we bind this?
    subject_embed_strategy: str = EmbedStrat.none # TODO update this once we consider mixed batches
    subject_embed_size: int = 256 # TODO can we bind this?
    task_embed_strategy: str = EmbedStrat.none # * we're not planning on going multitask in near future, so please hold.

    # This needs a separate API from the rest, likely, tied to readin.
    array_embed_strategy: str = EmbedStrat.none # ? subsumed by subject
    readin_strategy: str = EmbedStrat.project

    # Timestep level
    # "concat" becomes a valid strategy at this point
    stim_embed_strategy: str = EmbedStrat.token
    heldout_neuron_embed_strategy: str = EmbedStrat.token # Not even sure if there's a different way here.
    # There should maybe be a section for augmentation/ablation, but that is low pri.

class DataKey(Enum):
    # TODO need more thinking about this. Data is heterogenuous, can we maintain a single interface
    # when we have multiple arrays?
    spikes = 'spikes'
    stim = 'stim' # icms

class MetaKey(Enum):
    r"""
        Keys that are (potentially) tracked in `meta_df`
    """
    trial = 'trial'
    session = 'session'
    subject = 'subject'
    array = 'array'
    task = 'task'
    unique = 'unique' # default unique identifier

@dataclass
class DatasetConfig:
    root_dir: Path = Path("./data")
    preprocess_suffix: str = 'preprocessed'

    dataset_seed: int = 0 # for shuffling/splitting etc
    r"""
        Specifies the source dataset files (or potentially directories)
    """
    datasets: List[str] = []
    r"""
        `data_keys` and `meta_keys` specify the attributes of the dataset are served.
    """
    data_keys: List[DataKey] = [DataKey.spikes]
    meta_keys: List[MetaKey] = [MetaKey.session]

    split_key: MetaKey = MetaKey.unique
    # ==== Data parsing/processing ====
    bin_size_ms: int = 2
    pad_batches: bool = True # else, trim batches to the shortest trial
    max_trial_length: int = 1500 # in bins # TODO implement

    # TODO think about preprocessing strategies


@dataclass
class TrainConfig:
    epochs: int = 200
    steps: int = 0
    batch_size: int = 64
    patience: int = 500
    log_grad: bool = False
    gradient_clip_val: float = 0.0
    accumulate_batches: int = 1

@dataclass
class RootConfig:
    seed: int = 0
    tag: str = "" # i.e. experiment variant, now an optional tag (since hydra consumes file, we can't use the filename for experiment name. Specify if you want.)
    experiment_set: str = ""
    default_root_dir: Path = Path("./data/runs").resolve()
    wandb_project: str = "context_general_bci"
    wandb_api_key_path: Path = Path("/home/joelye/.wandb_api").resolve()
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()

    load_from_id: str = ""

    exp: Any = MISSING # delta config, provide via yaml and on CLI as `+exp=<test>.yaml`

cs = ConfigStore.instance()
cs.store(name="config", node=RootConfig)
