#%%
from typing import List, Union, Optional
from pathlib import Path
import math
import numpy as np
import torch
import torch.distributions as dists
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
# from scipy.signal import convolve
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

import logging
logger = logging.getLogger(__name__)
try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig, PittConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import compress_vector, heuristic_sanitize_payload

CLAMP_MAX = 15
NDT3_CAUSAL_SMOOTH_MS = 300
CURSOR_TRANSLATE_AND_CLICK = [1, 2, 6] # hardcoded dims for relevant xy + clikc dims from quicklogger raw
ALLOW_INTRA_TRIAL_CHOP = True
ALLOW_INTRA_TRIAL_CHOP = False

r"""
    Dev note to self: Pretty unclear how the .mat payloads we're transferring seem to be _smaller_ than n_element bytes. The output spike trials, ~250 channels x ~100 timesteps are reasonably, 25K. But the data is only ~10x this for ~100x the trials.
"""

def ReFIT(positions: torch.Tensor, goals: torch.Tensor, thresh: float = 0.01) -> torch.Tensor:
    magnitudes = torch.linalg.norm(velocities, dim=1)  # Compute magnitudes of original velocities
    angles = torch.atan2(velocities[:, 1], velocities[:, 0])  # Compute angles of velocities

    # Clip velocities with magnitudes below threshold to 0
    mask = (magnitudes < thresh)
    magnitudes[mask] = 0.0

    new_velocities = torch.stack((magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=1)

    return new_velocities



def extract_ql_data(ql_data):
    # ql_data: .mat['iData']['QL']['Data']
    # Currently just equipped to extract spike snippets
    # If you want more, look at `icms_modeling/scripts/preprocess_mat`
    # print(ql_data.keys())
    # print(ql_data['TASK_STATE_CONFIG'].keys())
    # print(ql_data['TASK_STATE_CONFIG']['state_num'])
    # print(ql_data['TASK_STATE_CONFIG']['state_name'])
    # print(ql_data['TRIAL_METADATA'])
    def extract_spike_snippets(spike_snippets):
        THRESHOLD_SAMPLE = 12./30000
        return {
            "spikes_source_index": spike_snippets['source_index'], # JY: I think this is NSP box?
            "spikes_channel": spike_snippets['channel'],
            "spikes_source_timestamp": spike_snippets['source_timestamp'] + THRESHOLD_SAMPLE,
            # "spikes_snippets": spike_snippets['snippet'], # for waveform
        }

    return {
        **extract_spike_snippets(ql_data['SPIKE_SNIPPET']['ss'])
    }

def events_to_raster(
    events,
    channels_per_array=128,
):
    """
        Tensorize sparse format.
    """
    events['spikes_channel'] = events['spikes_channel'] + events['spikes_source_index'] * channels_per_array
    bins = np.arange(
        events['spikes_source_timestamp'].min(),
        events['spikes_source_timestamp'].max(),
        0.001
    )
    timebins = np.digitize(events['spikes_source_timestamp'], bins, right=False) - 1
    spikes = torch.zeros((len(bins), 256), dtype=torch.uint8)
    spikes[timebins, events['spikes_channel']] = 1
    return spikes


def load_trial(fn, use_ql=True, key='data', copy_keys=True):
    # if `use_ql`, use the prebinned at 20ms and also pull out the kinematics
    # else take raw spikes
    # data = payload['data'] # 'data' is pre-binned at 20ms, we'd rather have more raw
    payload = loadmat(str(fn), simplify_cells=True, variable_names=[key] if use_ql else ['iData'])
    out = {
        'bin_size_ms': 20 if use_ql else 1,
        'use_ql': use_ql,
    }
    if use_ql:
        payload = payload[key]
        spikes = payload['SpikeCount']
        if spikes.shape[1] == 256 * 5:
            standard_channels = np.arange(0, 256 * 5,5) # unsorted, I guess
            spikes = spikes[..., standard_channels]
        out['spikes'] = torch.from_numpy(spikes)
        out['trial_num'] = torch.from_numpy(payload['trial_num'])
        if 'Kinematics' in payload:
            # cursor x, y
            out['position'] = torch.from_numpy(payload['Kinematics']['ActualPos'][:,1:3]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        elif 'pos' in payload:
            out['position'] = torch.from_numpy(payload['pos'][:,1:3]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        out['position'] = out['position'].roll(1, dims=1) # Pitt position logs in robot coords, i.e. y, dim 1 is up/down in cursor space, z, dim 2 is left/right in cursor space. Roll so we have x, y
        if 'target' in payload:
            out['target'] = torch.from_numpy(payload['target'][1:3].T) # dimensions flipped here, originally C x T
            out['target'] = out['target'].roll(1, dims=1) # Pitt position logs in robot coords, i.e. y, dim 1 is up/down in cursor space, z, dim 2 is left/right in cursor space. Roll so we have x, y
    else:
        data = payload['iData']
        trial_data = extract_ql_data(data['QL']['Data'])
        out['src_file'] = data['QL']['FileName']
        out['spikes'] = events_to_raster(trial_data)
    if copy_keys:
        for k in payload:
            if k not in out and k not in ['SpikeCount', 'trial_num', 'Kinematics', 'pos', 'target', 'QL', 'iData', 'data']:
                out[k] = payload[k]
    return out

def interpolate_nan(arr: Union[np.ndarray, torch.Tensor]):
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    out = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        x = arr[:, i]
        nans = np.isnan(x)
        non_nans = ~nans
        x_interp = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), x[non_nans])
        x[nans] = x_interp
        out[:, i] = x
    return torch.as_tensor(out)

@ExperimentalTaskRegistry.register
class PittCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.pitt_co

    @classmethod
    def get_kin_kernel(cls, causal_smooth_ms, sample_bin_ms=20) -> np.ndarray:
        kernel = np.ones((int(causal_smooth_ms / sample_bin_ms), 1), dtype=np.float32) / (causal_smooth_ms / sample_bin_ms)
        kernel[-kernel.shape[0] // 2:] = 0 # causal, including current timestep
        return kernel

    @staticmethod
    def smooth(position: Union[torch.Tensor, np.ndarray], kernel: np.ndarray) -> torch.Tensor:
        # kernel: e.g. =np.ones((int(180 / 20), 1))/ (180 / 20)
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        position = interpolate_nan(position)
        # position = position - position[0] # zero out initial position
        # Manually pad with edge values
        # OK to pad because this is beginning and end of _set_ where we expect little derivative (but possibly lack of centering)
        # assert kernel.shape[0] % 2 == 1, "Kernel must be odd (for convenience)"
        pad_left, pad_right = int(kernel.shape[0] / 2), int(kernel.shape[0] / 2)
        position = F.pad(position.T, (pad_left, pad_right), 'replicate')
        return F.conv1d(position.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1))[:,0].T

    @staticmethod
    def get_velocity(position, kernel, do_smooth=True):
        # kernel: np.ndarray, e.g. =np.ones((int(180 / 20), 1))/ (180 / 20)
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        if do_smooth:
            position = PittCOLoader.smooth(position.numpy().astype(dtype=kernel.dtype), kernel=kernel)
        else:
            position = interpolate_nan(position)
            position = torch.as_tensor(position)
        return torch.as_tensor(np.gradient(position.numpy(), axis=0), dtype=float) # note gradient preserves shape

    # @staticmethod
    # def get_velocity(position, kernel=np.ones((int(500 / 20), 2))/ (500 / 20)):
    #     # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
    #     # position = gaussian_filter1d(position, 2.5, axis=0) # This seems reasonable, but useless since we can't compare to Pitt codebase without below
    #     int_position = pd.Series(position.flatten()).interpolate()
    #     position = torch.tensor(int_position).view(-1, position.shape[-1])
    #     position = F.conv1d(position.T.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1), padding='same')[:,0].T
    #     vel = torch.as_tensor(np.gradient(position.numpy(), axis=0)).float() # note gradient preserves shape

    #     # position = pd.Series(position.flatten()).interpolate().to_numpy().reshape(-1, 2) # remove intermediate nans
    #     # position = convolve(position, kernel, mode='same')
    #     # vel = torch.tensor(np.gradient(position, axis=0)).float()
    #     # position = convolve(position, kernel, mode='same') # Nope. this applies along both dimensions. Facepalm.

    #     vel[vel.isnan()] = 0 # extra call to deal with edge values
    #     return vel

    @staticmethod
    def ReFIT(positions: torch.Tensor, goals: torch.Tensor, reaction_lag_ms=100, bin_ms=20, oracle_blend=0.25) -> torch.Tensor:
        # positions, goals: Time x Hidden.
        # weight: don't do a full refit correction, weight with original
        # defaults for lag experimented in `pitt_scratch`
        lag_bins = reaction_lag_ms // bin_ms
        empirical = PittCOLoader.get_velocity(positions)
        oracle = goals.roll(lag_bins, dims=0) - positions
        magnitudes = torch.linalg.norm(empirical, dim=1)  # Compute magnitudes of original velocities
        # Oracle magnitude update - no good, visually

        # angles = torch.atan2(empirical[:, 1], empirical[:, 0])  # Compute angles of velocities
        source_angles = torch.atan2(empirical[:, 1], empirical[:, 0])  # Compute angles of original velocities
        oracle_angles = torch.atan2(oracle[:, 1], oracle[:, 0])  # Compute angles of velocities

        # Develop a von mises update that blends the source and oracle angles
        source_concentration = 10.0
        oracle_concentration = source_concentration * oracle_blend

        # Create Von Mises distributions for source and oracle
        source_von_mises = dists.VonMises(source_angles, source_concentration)
        updated_angles = torch.empty_like(source_angles)

        # Mask for the nan values in oracle
        nan_mask = torch.isnan(oracle_angles)

        # Update angles where oracle is not nan
        if (~nan_mask).any():
            # Create Von Mises distributions for oracle where it's not nan
            oracle_von_mises = dists.VonMises(oracle_angles[~nan_mask], oracle_concentration)

            # Compute updated estimate as the circular mean of the two distributions.
            # We weight the distributions by their concentration parameters.
            updated_angles[~nan_mask] = (source_von_mises.concentration[~nan_mask] * source_von_mises.loc[~nan_mask] + \
                                        oracle_von_mises.concentration * oracle_von_mises.loc) / (source_von_mises.concentration[~nan_mask] + oracle_von_mises.concentration)

        # Use source angles where oracle is nan
        updated_angles[nan_mask] = source_angles[nan_mask]
        angles = updated_angles
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))

        new_velocities = torch.stack((magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=1)
        new_velocities[:reaction_lag_ms // bin_ms] = torch.nan  # We don't know what the goal is before the reaction lag, so we clip it
        # new_velocities[reaction_lag_ms // bin_ms:] = empirical[reaction_lag_ms // bin_ms:]  # Replace clipped velocities with original ones, for rolled time periods
        return new_velocities

    @classmethod
    def load(
        cls,
        datapath: Path, # path to matlab file
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ):
        assert cfg.bin_size_ms == 20, 'code not prepped for different resolutions'
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        def chop_vector(vec: torch.Tensor):
            # vec - already at target resolution, just needs chopping
            chop_size = round(cfg.pitt_co.chop_size_ms / cfg.bin_size_ms)
            return rearrange(
                vec.unfold(0, chop_size, chop_size),
                'trial hidden time -> trial time hidden'
             ) # Trial x C x chop_size (time)
        def save_trial_spikes(spikes, i, other_data={}):
            if DataKey.bhvr_vel in other_data:
                other_data[DataKey.bhvr_vel] = other_data[DataKey.bhvr_vel].float()
            single_payload = {
                DataKey.spikes: create_spike_payload(
                    spikes.clone(), arrays_to_use
                ),
                **other_data
            }
            # NDT3 parity, not used original NDT2 experimentss
            if not heuristic_sanitize_payload(single_payload):
                return
            single_path = cache_root / f'{dataset_alias}_{i}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        if (not datapath.is_dir() and datapath.suffix == '.mat') or str(datapath).endswith('.pth'): # payload style, preproc-ed/binned elsewhere
            if str(datapath).endswith('.pth'):
                payload = torch.load(datapath)
            else:
                payload = load_trial(datapath, key='thin_data')
            # Sanitize
            spikes = payload['spikes']
            # elements = spikes.nelement()
            unique, counts = np.unique(spikes, return_counts=True)
            for u, c in zip(unique, counts):
                if u >= CLAMP_MAX:
                    spikes[spikes == u] = CLAMP_MAX # clip
                # if u >= 15 or c / elements < 1e-5: # anomalous, suppress to max. (Some bins randomly report impossibly high counts like 90 (in 20ms))
                    # spikes[spikes == u] = 0

            # Iterate by trial, assumes continuity so we grab velocity outside
            # start_pad = round(500 / cfg.bin_size_ms)
            # end_pad = round(1000 / cfg.bin_size_ms)
            # should_clip = False
            exp_task_cfg: PittConfig = getattr(cfg, task.value)

            if (
                'position' in payload and \
                (task in [ExperimentalTask.observation, ExperimentalTask.ortho, ExperimentalTask.fbc, ExperimentalTask.unstructured] or \
                 str(datapath).endswith('.pth')
                 )  # and \ # Unstructured kinematics may be fake, mock data.
            ): # We only "trust" in the labels provided by obs (for now)
                if len(payload['position']) == len(payload['trial_num']):
                    if exp_task_cfg.closed_loop_intention_estimation == "refit" and task in [ExperimentalTask.ortho, ExperimentalTask.fbc]:
                        # breakpoint()
                        session_vel = PittCOLoader.ReFIT(payload['position'], payload['target'], bin_ms=cfg.bin_size_ms)
                    else:
                        session_vel = PittCOLoader.get_velocity(payload['position'], kernel=cls.get_kin_kernel(NDT3_CAUSAL_SMOOTH_MS, cfg.bin_size_ms))
                        if str(datapath).endswith('.pth'):
                            session_vel = session_vel[:,CURSOR_TRANSLATE_AND_CLICK]
                    # if session_vel[-end_pad:].abs().max() < 0.001: # likely to be a small bump to reset for next trial.
                    #     should_clip = True
                else:
                    session_vel = None
            else:
                session_vel = None
                
            if (
                'force' in payload
            ):
                # override! For grasp experiments
                session_vel = PittCOLoader.smooth(
                    payload['force'],
                    kernel=cls.get_kin_kernel(
                        NDT3_CAUSAL_SMOOTH_MS,
                        sample_bin_ms=cfg.bin_size_ms)
                    ) # Gary doesn't compute velocity, just absolute. We follow suit.
             
            # Add constraint logic for the sake of computing bhvr mask consistently with NDT3
            other_args = {}
            chop = 0 if exp_task_cfg.respect_trial_boundaries else exp_task_cfg.chop_size_ms
            sample_bin_ms = 20
            # clamp each constraint to 0 and 1 - otherwise nonsensical
            def cast_constraint(key: str) -> Optional[torch.Tensor]:
                vec: torch.Tensor | None = payload.get(key, None)
                if vec is None:
                    return None
                return vec.float().clamp(0, 1).half()
            brain_control = cast_constraint('brain_control')
            active_assist = cast_constraint('active_assist')
            passive_assist = cast_constraint('passive_assist')
            if brain_control is None or session_vel is None:
                chopped_constraints = None
            else:
                chopped_constraints = torch.stack([
                    compress_vector(1 - brain_control, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False), # return complement, such that native control is the "0" condition, no constraint
                    compress_vector(active_assist, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False),
                    compress_vector(passive_assist, chop_size_ms=chop, bin_size_ms=cfg.bin_size_ms, compression='max', sample_bin_ms=sample_bin_ms, keep_dim=False),
                ], -2) # Should be (Time x) Constraint x Domain. Note compress can outputs (chop x) time x domain, so stack -2 instead of forward. 2
                chopped_constraints = repeat(chopped_constraints, '... domain -> ... (domain 3)')[..., :session_vel.size(-1)] # Put behavioral control dimension last
                if DataKey.bhvr_mask in cfg.data_keys:
                    r"""
                        Behavior mask dictates when behavior targets are valid.
                        In open loop, this should be exactly when constraints are active (if we're not conditioning)
                        In closed loop, refit, this should be exactly when constraints are inactive, and brain control is on.
                        This key shouldn't be active if we're conditioning... TODO guards.
                    """
                    # constraint shape at this point is ... x 3 (constraint dim) x Behavior dim
                    if (chopped_constraints[..., 0, :] == 0).any(-1).any():
                        # ! Heuristic: At least 1 dimension has some brain control, suggests task phases are main ones where behavioral updates are relevant as opposed to AA
                        other_args[DataKey.bhvr_mask] = (chopped_constraints[..., 0, :] == 0).any(-1) 
                    # if exp_task_cfg.closed_loop_intention_estimation != "":
                        other_args[DataKey.bhvr_mask] = (chopped_constraints[..., 0, :] == 0).any(-1) # At least 1 dimension has some brain control # TODO need to address for partially assisted dimensions...
                    else:
                        other_args[DataKey.bhvr_mask] = (chopped_constraints[...,1, :] > 0).any(-1) # T # is active assist on ==> intention is on, labels are as valid as they will get
            
            if exp_task_cfg.respect_trial_boundaries and not task in [ExperimentalTask.unstructured]:
                for i in payload['trial_num'].unique():
                    if DataKey.bhvr_mask in cfg.data_keys and 'active_assist' in payload: # for NDT3
                        trial_mask = (payload['active_assist'] > 0).any(-1).unsqueeze(-1) # T x 1
                        trial_mask = trial_mask[payload['trial_num'] == i] # T x 1 -> T' x 1
                    else:
                        trial_mask = None
                    trial_spikes = payload['spikes'][payload['trial_num'] == i]
                    # trim edges -- typically a trial starts with half a second of inter-trial and ends with a second of failure/inter-trial pad
                    # we assume intent labels are not reliable in this timeframe
                    # if trial_spikes.size(0) <= start_pad + end_pad: # something's odd about this trial
                    #     continue
                    if session_vel is not None:
                        trial_vel = session_vel[payload['trial_num'] == i]
                    # if should_clip:
                    #     trial_spikes = trial_spikes[start_pad:-end_pad]
                    #     if session_vel is not None:
                    #         trial_vel = trial_vel[start_pad:-end_pad]
                    if trial_spikes.size(0) == 0:
                        continue
                    if trial_spikes.size(0) < round(exp_task_cfg.chop_size_ms / cfg.bin_size_ms) or not ALLOW_INTRA_TRIAL_CHOP:
                        other_args_trial = {DataKey.bhvr_vel: trial_vel} if session_vel is not None else {}
                        if trial_mask is not None:
                            other_args_trial[DataKey.bhvr_mask] = trial_mask
                        save_trial_spikes(trial_spikes, i, other_args_trial)
                    else:
                        chopped_spikes = chop_vector(trial_spikes)
                        if session_vel is not None:
                            chopped_vel = chop_vector(trial_vel)
                        if trial_mask is not None:
                            chopped_mask = chop_vector(trial_mask)
                        for j, subtrial_spikes in enumerate(chopped_spikes):
                            other_args_trial = {DataKey.bhvr_vel: chopped_vel[j]} if session_vel is not None else {}
                            if trial_mask is not None:
                                other_args_trial[DataKey.bhvr_mask] = chopped_mask[j]
                            save_trial_spikes(subtrial_spikes, f'{i}_trial{j}', other_args_trial)

                        end_of_trial = trial_spikes.size(0) % round(exp_task_cfg.chop_size_ms / cfg.bin_size_ms)
                        if end_of_trial > 0:
                            trial_spikes_end = trial_spikes[-end_of_trial:]
                            if session_vel is not None:
                                trial_vel_end = trial_vel[-end_of_trial:]
                            if trial_mask is not None:
                                trial_mask_end = trial_mask[-end_of_trial:]
                            other_args_trial = {DataKey.bhvr_vel: trial_vel_end} if session_vel is not None else {}
                            if trial_mask is not None:
                                other_args_trial[DataKey.bhvr_mask] = trial_mask_end
                            save_trial_spikes(trial_spikes_end, f'{i}_end', other_args_trial)
            else:
                # chop both
                spikes = chop_vector(spikes)
                if session_vel is not None:
                    session_vel = chop_vector(session_vel)
                for i, trial_spikes in enumerate(spikes):
                    other_args_trial = {DataKey.bhvr_vel: session_vel[i]} if session_vel is not None else {}
                    for k in other_args:
                        other_args_trial[k] = other_args[k][i]
                    save_trial_spikes(trial_spikes, i, {DataKey.bhvr_vel: session_vel[i]} if session_vel is not None else {})
        else: # folder style, preproc-ed on mind
            for i, fname in enumerate(datapath.glob("*.mat")):
                if fname.stem.startswith('QL.Task'):
                    payload = load_trial(fname)
                    single_payload = {
                        DataKey.spikes: create_spike_payload(
                            payload['spikes'], arrays_to_use, cfg, payload['bin_size_ms']
                        ),
                    }
                    if 'position' in payload:
                        single_payload[DataKey.bhvr_vel] = PittCOLoader.get_velocity(payload['position'])
                    single_path = cache_root / f'{i}.pth'
                    meta_payload['path'].append(single_path)
                    torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)


# Register aliases
ExperimentalTaskRegistry.register_manual(ExperimentalTask.observation, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.ortho, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.fbc, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.unstructured, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.pitt_co, PittCOLoader)