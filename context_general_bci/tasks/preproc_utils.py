from typing import Union, Dict
import numpy as np
import torch
from einops import rearrange, reduce

from context_general_bci.config import DataKey

def compress_vector(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int, compression='sum', sample_bin_ms=1, keep_dim=True):
    r"""
        # vec: at sampling resolution of 1ms, T C. Useful for things that don't have complicated downsampling e.g. spikes.
        # chop_size_ms: chop size in ms. If 0, doesn't chop
        # bin_size_ms: bin size in ms - target bin size, after comnpression
        # sample_bin_ms: native res of vec
        Crops tail if not divisible by bin_size_ms
    """

    if chop_size_ms:
        if vec.size(0) < chop_size_ms // sample_bin_ms:
            # No extra chop needed, just directly compress
            full_vec = vec.unsqueeze(0)
            # If not divisible by subsequent bin, crop
            if full_vec.shape[1] % (bin_size_ms // sample_bin_ms) != 0:
                full_vec = full_vec[:, :-(full_vec.shape[1] % (bin_size_ms // sample_bin_ms)), :]
            full_vec = rearrange(full_vec, 'b time c -> b c time')
        else:
            full_vec = vec.unfold(0, chop_size_ms // sample_bin_ms, chop_size_ms // sample_bin_ms) # Trial x C x chop_size (time)
        full_vec = rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'b time c 1' if keep_dim else 'b time c'
            return reduce(full_vec, f'b time c bin -> {out_str}', compression)
        if keep_dim:
            return full_vec[..., -1:]
        return full_vec[..., -1]
    else:
        if vec.shape[0] % (bin_size_ms // sample_bin_ms) != 0:
            vec = vec[:-(vec.shape[0] % (bin_size_ms // sample_bin_ms))]
        vec = rearrange(vec, '(time bin) c -> time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'time c 1' if keep_dim else 'time c'
            return reduce(vec, f'time c bin -> {out_str}', compression)
        if keep_dim:
            return vec[..., -1:]
        return vec[..., -1]


def heuristic_sanitize(spikes: Union[torch.Tensor, Dict[str, torch.Tensor]], behavior) -> bool:
    r"""
        Given spike and behavior arrays, apply heuristics to tell whether data is valid.
        Assumes data is binned at 20ms
            spikes: Time x Neurons ...
            behavior: Time x Bhvr Dim ...
    """
    if isinstance(spikes, dict):
        all_spike_sum = 0
        for k, v in spikes.items():
            if v.shape[0] < 5:
                return False
            all_spike_sum += v.sum()
        if all_spike_sum == 0:
            return False
    else:
        if spikes.shape[0] < 5: # Too short, reject.
            return False
        if spikes.sum() == 0:
            return False
    # check if behavior is constant
    if behavior is not None:
        if isinstance(behavior, torch.Tensor):
            if torch.isclose(behavior.std(0), torch.tensor(0.,)).all():
                return False
        else:
            if np.isclose(behavior.std(0), 0).all():
                return False
    return True

def heuristic_sanitize_payload(payload: Dict[DataKey, Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> bool:
    return heuristic_sanitize(payload[DataKey.spikes], payload.get(DataKey.bhvr_vel, None))
