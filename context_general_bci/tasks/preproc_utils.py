from typing import Union, List, Dict, TypeVar
import numpy as np
import torch
from einops import rearrange, reduce

from context_general_bci.config import DataKey

T = TypeVar('T', torch.Tensor, None)

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

def compress_vector_ndt3(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int, compression='sum', sample_bin_ms=1, keep_dim=True):
    r"""
        Exact copy paste of NDT3 code for parity.
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


def chop_vector(vec: T, chop_size_ms: int, bin_size_ms: int) -> T:
    # vec - T H
    # vec - already at target resolution, just needs chopping. e.g. useful for covariates that have been externally downsampled
    if chop_size_ms == 0:
        return vec
    if vec is None:
        return None
    chops = round(chop_size_ms / bin_size_ms)
    if vec.size(0) <= chops:
        return rearrange(vec, 'time hidden -> 1 time hidden')
    else:
        return rearrange(
            vec.unfold(0, chops, chops),
            'trial hidden time -> trial time hidden'
            ) # Trial x C x chop_size (time)

def spike_times_to_dense(spike_times_ms: List[Union[np.ndarray, np.float64, np.int32]], bin_size_ms: int, time_start=0, time_end=0, speculate_start=False) -> torch.Tensor:
    # spike_times_ms: List[Channel] of spike times, in ms from trial start
    # return: Time x Channel x 1, at bin resolution
    # Create at ms resolution
    for i in range(len(spike_times_ms)):
        if len(spike_times_ms[i].shape) == 0:
            spike_times_ms[i] = np.array([spike_times_ms[i]]) # add array dim
    time_flat = np.concatenate(spike_times_ms)
    if time_end == 0:
        time_end = time_flat.max()
    else:
        spike_times_ms = [s[s < time_end] if s is not None else s for s in spike_times_ms]
    if time_start == 0 and speculate_start: # speculate was breaking change
        speculative_start = time_flat.min()
        if time_end - speculative_start < speculative_start: # If range of times is smaller than start point, clock probably isn't zeroed out
            # print(f"Spike time speculative start: {speculative_start}, time_end: {time_end}")
            time_start = speculative_start

    dense_bin_count = math.ceil(time_end - time_start)
    if time_start != 0:
        spike_times_ms = [s[s >= time_start] - time_start if s is not None else s for s in spike_times_ms]

    trial_spikes_dense = torch.zeros(len(spike_times_ms), dense_bin_count, dtype=torch.uint8)
    for channel, channel_spikes_ms in enumerate(spike_times_ms):
        if channel_spikes_ms is None or len(channel_spikes_ms) == 0:
            continue
        # Off-by-1 clip
        channel_spikes_ms = np.minimum(np.floor(channel_spikes_ms), trial_spikes_dense.shape[1] - 1)
        trial_spikes_dense[channel] = torch.bincount(torch.as_tensor(channel_spikes_ms, dtype=torch.int), minlength=trial_spikes_dense.shape[1])
    trial_spikes_dense = trial_spikes_dense.T # Time x Channel
    return compress_vector_ndt3(trial_spikes_dense, 0, bin_size_ms)
