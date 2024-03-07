from typing import Dict
import logging
import torch
from context_general_bci.config import DataKey, BatchKey
from context_general_bci.dataset import (
    LENGTH_KEY,
    CHANNEL_KEY,
    COVARIATE_LENGTH_KEY,
    COVARIATE_CHANNEL_KEY
)

r"""
    Data utilities for mixing streams, should be model agnostic
"""

def precrop_batch(
    batch: Dict[BatchKey, torch.Tensor], # item also works (no batch dimension), due to broadcasting
    crop_timesteps: int,
):
    r"""
        Keep timestep to < crop_timesteps
    """
    sanitize = lambda x: x.name if DataKey.time.name in batch else x # stringify - needed while we have weird dataloader misparity
    spike_time = batch[sanitize(DataKey.time)]

    flatten = spike_time.ndim == 2
    if flatten:
        if spike_time.shape[0] > 1:
            logging.warning(f"Assuming consistent time across batch ({spike_time.shape[0]})")
        spike_time = spike_time[0]
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][:, spike_time < crop_timesteps],
            sanitize(DataKey.time): batch[sanitize(DataKey.time)][:, spike_time < crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][:, spike_time < crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:, :crop_timesteps],
            sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][:, :crop_timesteps],
            CHANNEL_KEY: batch[CHANNEL_KEY][:, spike_time < crop_timesteps],
        }
    else:
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][spike_time < crop_timesteps],
            sanitize(DataKey.time): spike_time[spike_time < crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][spike_time < crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:crop_timesteps],
            sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][:crop_timesteps],
            CHANNEL_KEY: batch[CHANNEL_KEY][spike_time < crop_timesteps],
        }
    for k in batch:
        if k in out:
            continue
        if k == COVARIATE_CHANNEL_KEY:
            continue
            raise NotImplementedError("Covariate channel not supported")
        if k == COVARIATE_LENGTH_KEY: # Nonspatial
            out[k] = torch.min(torch.tensor(crop_timesteps), batch[k])
        elif k == LENGTH_KEY: # Spatial
            out[k] = torch.min(torch.tensor(out[DataKey.spikes].shape[-3]), batch[k]) # Min of crop_timesteps and original length
        else:
            out[k] = batch[k]
    return out

def postcrop_batch(
    batch: Dict[BatchKey, torch.Tensor],
    crop_timesteps: int,
):
    r"""
        Take suffix crop by ABSOLUTE crop_timesteps, >= given timestep ! NOT number of timesteps.
    """
    # ! In place mod
    # Hm. This will flatten the batch, since there's no guarantees. OK, we'll just squeeze out the time dimension
    sanitize = lambda x: x.name if x.name in batch else x  # stringify
    spike_time = batch[sanitize(DataKey.time)]
    flatten = spike_time.ndim == 2
    if flatten:
        if spike_time.shape[0] > 1:
            logging.warning(f"Assuming consistent time across batch ({spike_time.shape[0]})")
        spike_time = spike_time[0]
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][:, spike_time >= crop_timesteps],
            sanitize(DataKey.time): batch[sanitize(DataKey.time)][:, spike_time >= crop_timesteps] - crop_timesteps,
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][:, spike_time >= crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][:, crop_timesteps:],
            sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][:, crop_timesteps:],
            CHANNEL_KEY: batch[CHANNEL_KEY][:, spike_time >= crop_timesteps],
        }
    else:
        out = {
            sanitize(DataKey.spikes): batch[sanitize(DataKey.spikes)][spike_time >= crop_timesteps],
            sanitize(DataKey.time): spike_time[spike_time >= crop_timesteps],
            sanitize(DataKey.position): batch[sanitize(DataKey.position)][spike_time >= crop_timesteps],
            sanitize(DataKey.bhvr_vel): batch[sanitize(DataKey.bhvr_vel)][crop_timesteps:],
            sanitize(DataKey.bhvr_mask): batch[sanitize(DataKey.bhvr_mask)][crop_timesteps:],
            CHANNEL_KEY: batch[CHANNEL_KEY][spike_time >= crop_timesteps],
        }
    for k in batch:
        if k in out:
            continue
        if k == COVARIATE_CHANNEL_KEY:
            continue
            raise NotImplementedError("Covariate channel not supported")
        if k == COVARIATE_LENGTH_KEY: # Nonspatial - what if we have padding at end? Cropping doesn't really make a lot of sense...
            out[k] = torch.max(batch[k] - crop_timesteps, torch.tensor(0))
        elif k == LENGTH_KEY: # Spatial
            out[k] = torch.max(batch[k] - (batch[DataKey.spikes].shape[-3] - out[DataKey.spikes].shape[-3]), torch.tensor(0))
        else:
            out[k] = batch[k]
    return out
