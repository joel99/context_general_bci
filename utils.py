# Miscellany
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import wandb
import torch

def get_newest_ckpt_in_dir(ckpt_dir: Path):
    # Newest is best since we have early stopping callback, and modelcheckpoint only saves early stopped checkpoints (not e.g. latest)
    return sorted(ckpt_dir.glob("*.ckpt"), key=osp.getmtime)[-1]

def get_latest_ckpt_from_wandb_id(
        wandb_project, wandb_id, wandb_user="joelye9"
    ):
    wandb_id = wandb_id.split('-')[-1]
    api = wandb.Api()
    run = api.run(f"{wandb_user}/{wandb_project}/{wandb_id}")
    ckpt_dir = Path(wandb_project) / wandb_id / "checkpoints" # curious, something about checkpoint dumping isn't right
    return get_newest_ckpt_in_dir(ckpt_dir)
