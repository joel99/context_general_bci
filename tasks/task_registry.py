from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from config import DatasetConfig

r"""
    Super light wrapper to define task/loader interface.
    # TODO make into an actual registry
"""
class ExperimentalTaskLoader:
    @classmethod
    def load(cls, path: Path, cfg: DatasetConfig, cache_root: Path) -> pd.DataFrame:
        r"""
            Load data from `path` into a dataframe.
            `cfg` contains information about how to load the data.
            (and should contain a sub-config corresponding to any registered task)
            `cache_root` is the root directory for caching single trials.

            Each loader should be responsible for loading/caching all information in paths
        """
        raise NotImplementedError