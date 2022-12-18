# Miscellany
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
from typing import get_type_hints, get_args
from collections import defaultdict
from enum import Enum
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import wandb
import scipy.signal as signal
import torch
import itertools
from dacite import from_dict

from model import BrainBertInterface, load_from_checkpoint
from data import DataAttrs
from config import RootConfig


# Wandb management

def cast_paths_and_enums(cfg: Dict, template=RootConfig()):
    # recursively cast any cfg field that is a path in template to a path, since dacite doesn't support our particular case quite well
    # thinking about it more - the weak link is wandb; which casts enums and paths to __str__
    # and now we have to recover from __str__
    def search_enum(str_rep: str, enum: Enum):
        for member in enum:
            if str_rep == str(member):
                return member
        raise ValueError(f"Could not find {str_rep} in {enum}")
    for k, v in get_type_hints(template).items():
        if v == Any: # optional values
            continue
        elif k not in cfg:
            continue # new attr
        elif v == Path:
            cfg[k] = Path(cfg[k])
        elif isinstance(cfg[k], list):
            for i, item in enumerate(cfg[k]):
                generic = get_args(v)[0]
                if issubclass(generic, Enum):
                    cfg[k][i] = search_enum(item, generic)
        elif issubclass(v, Enum):
            cfg[k] = search_enum(cfg[k], v)
        elif isinstance(cfg[k], dict):
            # print(f"recursing with {k}")
            cast_paths_and_enums(cfg[k], template=v)
    return cfg

def create_typed_cfg(cfg: Dict) -> RootConfig:
    cfg = cast_paths_and_enums(cfg)
    return from_dict(data_class=RootConfig, data=cfg)


WandbRun = Any

def load_wandb_run(run: WandbRun, tag="val-") -> Tuple[BrainBertInterface, RootConfig, DataAttrs]:
    run_data_attrs = from_dict(data_class=DataAttrs, data=run.config['data_attrs'])
    del run.config['data_attrs']
    cfg: RootConfig = OmegaConf.create(create_typed_cfg(run.config)) # Note, unchecked cast, but we often fiddle with irrelevant variables and don't want to get caught up
    ckpt = get_latest_ckpt_from_wandb_id(cfg.wandb_project, run.id, tag=tag)
    print(f"Loading {ckpt}")
    model = load_from_checkpoint(ckpt)
    # model = BrainBertInterface.load_from_checkpoint(ckpt, cfg=cfg)
    return model, cfg, run_data_attrs

def get_newest_ckpt_in_dir(ckpt_dir: Path, tag="val-"):
    # Newest is best since we have early stopping callback, and modelcheckpoint only saves early stopped checkpoints (not e.g. latest)
    res = sorted(ckpt_dir.glob("*.ckpt"), key=osp.getmtime)
    res = [r for r in res if tag in r.name]
    return res[-1]

def get_latest_ckpt_from_wandb_id(
        wandb_project,
        wandb_id,
        tag = ""
    ):
    wandb_id = wandb_id.split('-')[-1]
    ckpt_dir = Path(wandb_project) / wandb_id / "checkpoints" # curious, something about checkpoint dumping isn't right
    return get_newest_ckpt_in_dir(ckpt_dir, tag=tag)

def get_wandb_run(wandb_id, wandb_project='context_general_bci', wandb_user="joelye9"):
    wandb_id = wandb_id.split('-')[-1]
    api = wandb.Api()
    return api.run(f"{wandb_user}/{wandb_project}/{wandb_id}")


# TODO update these
def wandb_query_latest(
    name_kw,
    wandb_user='joelye9',
    wandb_project='context_general_bci',
    exact=False,
    allow_running=False,
    **filter_kwargs
) -> List[Any]: # returns list of wandb run objects
    # One can imagine moving towards a world where we track experiment names in config and query by experiment instead of individual variants...
    # But that's for the next project...
    # Default sort order is newest to oldest, which is what we want.
    api = wandb.Api()
    target = name_kw if exact else {"$regex": name_kw}
    states = ["finished", "crashed"]
    if allow_running:
        states.append("running")
    filters = {
        # "display_name": Target,
        "config.tag": target,
        "state": {"$in": states}, # crashed == timeout
        **filter_kwargs
    }
    runs = api.runs(
        f"{wandb_user}/{wandb_project}",
        filters=filters
    )
    return runs

def wandb_query_several(
    strings,
    min_time=None,
    latest_for_each_seed=True,
):
    runs = []
    for s in strings:
        runs.extend(wandb_query_latest(
            s, exact=True, latest_for_each_seed=latest_for_each_seed,
            created_at={
                "$gt": min_time if min_time else "2022-01-01"
                }
            ,
            allow_running=True # ! NOTE THIS
        ))
    return runs

def stack_batch(batch_out: List[Dict[str, torch.Tensor]]):
    all_lists = defaultdict(list)
    for batch in batch_out:
        for k, v in batch.items():
            all_lists[k].extend(v)
    out = {}
    for k, v in all_lists.items():
        if isinstance(v[0], torch.Tensor):
            # try stack
            if all(v2.size() == v[0].size() for v2 in v[1:]):
                out[k] = torch.stack(v)
            else:
                out[k] = v
        else:
            out[k] = torch.tensor(v).mean() # some metric
    return out


# Compute Gauss window and std with respect to bins
def gauss_smooth(spikes, bin_size, kernel_sd=0.05):
    # input b t c
    gauss_bin_std = kernel_sd / bin_size
    # the window extends 3 x std in either direction
    win_len = int(7 * gauss_bin_std) # ! Changed from 6 to 7 so there is a peak
    # Create Gaussian kernel
    window = signal.gaussian(win_len, gauss_bin_std, sym=True)
    window /=  np.sum(window)
    window = torch.tensor(window, dtype=torch.float)
    # Record B T C
    b, t, c = spikes.size()
    spikes = spikes.permute(0, 2, 1).reshape(b*c, 1, t)
    # Convolve window (B 1 T) with record as convolution will sum across channels.
    window = window.unsqueeze(0).unsqueeze(0)
    smooth_spikes = torch.nn.functional.conv1d(spikes, window, padding="same")
    return smooth_spikes.reshape(b, c, t).permute(0, 2, 1)


def prep_plt(ax=None, **kwargs):
    if isinstance(ax, np.ndarray):
        for _ax in ax.ravel():
            _prep_plt(_ax, **kwargs)
    else:
        ax = _prep_plt(ax, **kwargs)
    return ax

def _prep_plt(ax=None, spine_alpha=0.3, big=False):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    LARGE_SIZE = 15
    if big:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 24
        LARGE_SIZE = 28
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    if ax is None:
        plt.figure(figsize=(6,4))
        ax = plt.gca()
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    # ax.spines['top'].set_alpha(spine_alpha)
    ax.spines['top'].set_alpha(0)
    # ax.spines['right'].set_alpha(spine_alpha)
    ax.spines['right'].set_alpha(0)
    ax.grid(alpha=0.25)
    return ax


