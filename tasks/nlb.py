from typing import Dict, Any
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch

import logging

from config import DataKey, MetaKey, DatasetConfig
from context_registry import context_registry
from subjects import SubjectArrayRegistry, SubjectName, ArrayID
from tasks.task_registry import ExperimentalTaskLoader
TrialNum = int
MetadataKey = str

class NLBLoader(ExperimentalTaskLoader):
    def load(path: Path, cfg: DatasetConfig, cache_root: Path):
        r"""
            Loader for motor tasks in Neural Latents Benchmark (NLB) dataset.
        """
        # TODO
        return pd.DataFrame({})