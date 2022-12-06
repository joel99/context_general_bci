from typing import Dict, List
from pathlib import Path
import pandas as pd

from config import DatasetConfig
from subjects import SubjectInfo
r"""
    Super light wrapper to define task/loader interface.
"""
class ExperimentalTaskLoader:
    name: str
    @classmethod
    def load(cls,
        dataset_path: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        arrays: List[str],
        dataset_alias: str,
    ) -> pd.DataFrame:
        r"""
            Load data from `path` into a dataframe.
            `cfg` contains information about how to load the data.
            (and should contain a sub-config corresponding to any registered task)
            `cache_root` is the root directory for caching single trials.

            Each loader should be responsible for loading/caching all information in paths
        """
        raise NotImplementedError


class ExperimentalTaskRegistry:
    _instance = None
    _loaders: Dict[str, ExperimentalTaskLoader] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, to_register: ExperimentalTaskLoader):
        def wrap(to_register: ExperimentalTaskLoader):
            cls._loaders[to_register.name] = to_register
            return to_register
        return wrap(to_register)

    @classmethod
    def get_loader(cls, name: str) -> ExperimentalTaskLoader:
        return cls._loaders[name]