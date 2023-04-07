from hydra.core.config_store import ConfigStore
from .config_base import *
import config.presets

cs = ConfigStore.instance()
cs.store(name="config", node=RootConfig)