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
    task: ModelTask = ModelTask.infill


@dataclass
class PretrainingModelConfig(ModelConfig):
    task: TaskConfig = field(default_factory=InfillTaskConfig)
    lr_init: float = 5e-4
    lr_ramp_steps: int = 50
    lr_decay_steps: int = 1000
cs.store(group="model", name="pretrain", node=PretrainingModelConfig)

@dataclass
class FinetuningModelConfig(ModelConfig):
    lr_init: float = 2e-5
    lr_ramp_steps: int = 500 # epochs tend to be small
    lr_decay_steps: int = 5000 # epochs tend to be small

cs.store(group="model", name="finetune", node=FinetuningModelConfig)

@dataclass
class NLBModelTaskConfig(TaskConfig):
    task: ModelTask = ModelTask.heldout_decoding
    metrics: List[Metric] = field(default_factory=lambda: [Metric.co_bps, Metric.block_co_bps])
    outputs: List[Output] = field(default_factory=lambda: [Output.heldout_logrates])

@dataclass
class NLBModelConfig(FinetuningModelConfig):
    task: TaskConfig = field(default_factory=NLBModelTaskConfig)

cs.store(group="model", name="nlb", node=NLBModelConfig)

@dataclass
class NLBTrainConfig(TrainConfig):
    epochs: int = 50000 # epochs tend to be small
    batch_size: int = 128
    patience: int = 500

cs.store(group="train", name="nlb", node=NLBTrainConfig)

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
