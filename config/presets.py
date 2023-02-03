from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

r"""
    In Hydra, experimental presets can be declared either in YAML, or via the ConfigStore API.
    We will use ConfigStore API for the type safety.
"""

from .config_base import *

cs = ConfigStore.instance()

@dataclass
class InfillTaskConfig(TaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.infill])

@dataclass
class SmallTransformerConfig(TransformerConfig):
    n_state: int = 128
    dropout: float = 0.6

@dataclass
class SmallTransformerConfigLessDrop(TransformerConfig):
    n_state: int = 128
    dropout: float = 0.3


@dataclass
class PretrainingModelConfig(ModelConfig):
    r"""
        BERT does 10K ramp, 1M full. We are ~2 orders of magnitude data smaller.
        Our "steps" below are in epochs - so adjust appropriately. Currently
        ~10 batches per epoch in 2K trials (unaggregate)
    """
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 500
    lr_decay_steps: int = 10000
    dropout: float = 0.3
    hidden_size: int = 128
    session_embed_size: int = 128
    subject_embed_size: int = 128
    array_embed_size: int = 128
    transformer: TransformerConfig = field(default_factory=SmallTransformerConfigLessDrop)
cs.store(group="model", name="pretrain", node=PretrainingModelConfig)

@dataclass
class PretrainingNewModelConfig(ModelConfig):
    # A little more informed after initial experimentation
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 100
    lr_decay_steps: int = 1000
    dropout: float = 0.1
    hidden_size: int = 256
    session_embed_size: int = 256
    subject_embed_size: int = 256
    array_embed_size: int = 256
    # base config: 6 layers, 256 hidden, 4 heads
cs.store(group="model", name="pretrain_2x", node=PretrainingNewModelConfig)

@dataclass
class PretrainingSmallModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_ramp_steps: int = 3000
    lr_decay_steps: int = 100000
    dropout: float = 0.6
    hidden_size: int = 128
    session_embed_size: int = 128
    subject_embed_size: int = 128
    array_embed_size: int = 128
    transformer: TransformerConfig = field(default_factory=SmallTransformerConfig)

cs.store(group="model", name="pretrain_small", node=PretrainingSmallModelConfig)

@dataclass
class FinetuningModelConfig(ModelConfig):
    lr_init: float = 5e-5
    lr_ramp_steps: int = 3000 # epochs tend to be small
    lr_decay_steps: int = 100000 # epochs tend to be small

cs.store(group="model", name="finetune", node=FinetuningModelConfig)

@dataclass
class NLBModelTaskConfig(TaskConfig):
    tasks: List[ModelTask] = field(default_factory=lambda: [ModelTask.heldout_decoding])
    metrics: List[Metric] = field(default_factory=lambda: [Metric.co_bps, Metric.block_co_bps])
    outputs: List[Output] = field(default_factory=lambda: [Output.heldout_logrates])

cs.store(group='model/task', name='nlb', node=NLBModelTaskConfig)
@dataclass
class NLBModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=NLBModelTaskConfig)

cs.store(group="model", name="nlb", node=NLBModelConfig)

@dataclass
class PretrainConfig(TrainConfig):
    epochs: int = 4000
    batch_size: int = 128
    patience: int = 50
# As we hit >10K datapts, we typically see convergence in ~800 epochs at most.
cs.store(group="train", name="pretrain", node=PretrainConfig)

@dataclass
class NLBTrainConfig(TrainConfig):
    epochs: int = 50000 # epochs tend to be small
    batch_size: int = 64
    patience: int = 2000
    # patience: int = 4000

cs.store(group="train", name="nlb", node=NLBTrainConfig)
cs.store(group="train", name="small", node=NLBTrainConfig) # alias

@dataclass
class RTTDataConfig(DatasetConfig):
    r"""
        Default configuration for all RTT datasets
    """
    bin_size_ms: int = 20 # 20 is the new king...
    # bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt', 'odoherty_rtt.*'])
    max_channels: int = 98
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes])
    max_arrays: int = 1
    meta_keys: List[MetaKey] = field(default_factory=lambda: [
        MetaKey.unique, MetaKey.array, MetaKey.subject, MetaKey.session,
    ])

@dataclass
class RTTNLBDataConfig(DatasetConfig):
    r"""
        Default configuration for RTT NLB fine-tuning
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt'])
    max_channels: int = 98
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])

cs.store(group="dataset", name="rtt_nlb", node=RTTNLBDataConfig)

@dataclass
class MCMazeExpConfig(NLBConfig):
    heldout_neurons: int = 45

cs.store(group='dataset/nlb_maze', name='mc_maze', node=MCMazeExpConfig)

@dataclass
class MazeDataConfig(DatasetConfig):
    r"""
        Default configuration for all maze datasets
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_maze.*', 'churchland_maze_*'])
    max_channels: int = 137
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes])
    nlb_maze: NLBConfig = field(default_factory=MCMazeExpConfig)
    max_arrays: int = 2
    meta_keys: List[MetaKey] = field(default_factory=lambda: [
        MetaKey.unique, MetaKey.array, MetaKey.subject, MetaKey.session,
    ])
cs.store(group="dataset", name="maze", node=MazeDataConfig)

@dataclass
class MazeNLBDataConfig(DatasetConfig):
    r"""
        Default configuration for NLB-dataset pretrain or fine-tuning
    """
    bin_size_ms: int = 5
    datasets: List[str] = field(default_factory=lambda: ['mc_maze$'])
    max_channels: int = 137
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])
    nlb_maze: NLBConfig = field(default_factory=MCMazeExpConfig)

cs.store(group="dataset", name="maze_nlb", node=MazeNLBDataConfig)

@dataclass
class RTTExpConfig(NLBConfig):
    heldout_neurons: int = 32

@dataclass
class ODohertyExpConfig(RTTConfig):
    arrays: List[str] = field(default_factory=lambda: ['Indy-M1', 'Loco-M1'])

cs.store(group='dataset/nlb_rtt', name='mc_rtt', node=RTTExpConfig)

@dataclass
class RTTDataConfig(DatasetConfig):
    r"""
        Default configuration for all maze datasets NLB fine-tuning
    """
    bin_size_ms: int = 20
    datasets: List[str] = field(default_factory=lambda: ['mc_rtt', 'odoherty_rtt.*'])
    max_channels: int = 98
    max_arrays: int = 1
    data_keys: List[DataKey] = field(default_factory=lambda: [DataKey.spikes, DataKey.heldout_spikes])
    nlb_rtt: NLBConfig = field(default_factory=RTTExpConfig)
    odoherty_rtt: ODohertyExpConfig = field(default_factory=ODohertyExpConfig)

cs.store(group="dataset", name="rtt", node=RTTDataConfig)
