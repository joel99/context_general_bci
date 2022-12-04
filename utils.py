# Miscellany
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import wandb
import scipy.signal as signal
import torch

# Some types
StimCommand = NamedTuple("StimCommand", times=np.ndarray, channels=np.ndarray, current=np.ndarray)
CommandPayload = Dict[Path, StimCommand]

def get_newest_ckpt_in_dir(ckpt_dir: Path):
    # Newest is best since we have early stopping callback, and modelcheckpoint only saves early stopped checkpoints (not e.g. latest)
    return sorted(ckpt_dir.glob("*.ckpt"), key=osp.getmtime)[-1]

def get_latest_ckpt_from_wandb_id(
        wandb_project, wandb_id, wandb_user="joelye9"
    ):
    wandb_id = wandb_id.split('-')[-1]
    ckpt_dir = Path(wandb_project) / wandb_id / "checkpoints" # curious, something about checkpoint dumping isn't right
    return get_newest_ckpt_in_dir(ckpt_dir)

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


