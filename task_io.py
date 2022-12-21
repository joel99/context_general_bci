from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack # baby steps...

import logging

from config import (
    ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey,
)

from data import DataAttrs, LENGTH_KEY, CHANNEL_KEY, HELDOUT_CHANNEL_KEY
from subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)

class TaskPipeline(nn.Module):
    r"""
        Task IO - manages decoder layers, loss functions
        i.e. is responsible for returning loss, decoder outputs, and metrics
        # TODO manage input
        # TODO manage additional metrics
    """
    does_update_root = False

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ) -> None:
        super().__init__()

    def get_temporal_context(self, batch: Dict[str, torch.Tensor]):
        r"""
            For task specific temporal _input_. (B T H)
        """
        return []

    def get_trial_context(self, batch: Dict[str, torch.Tensor]):
        r"""
            For task specific trial _input_. (B H)
        """
        raise NotImplementedError # Nothing in main model to use this
        return []

    def update_batch(self, batch: Dict[str, torch.Tensor]):
        r"""
            Currently redundant with get_temporal_context - need to refactor.
            It could be that this forces a one-time modification.
            Update batch in place for modifying/injecting batch info.
        """
        return batch

    def get_trial_query(self, batch: Dict[str, torch.Tensor]):
        r"""
            For task specific trial _query_. (B H)
        """
        raise NotImplementedError # nothing in main model to use this
        return []

    def forward(self, batch, backbone_features: torch.Tensor, compute_metrics=True) -> torch.Tensor:
        r"""
            If compute_metrics is False, only return outputs. (Typically used in inference)
            Else also log metrics.
        """
        raise NotImplementedError

class RatePrediction(TaskPipeline):
    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        decoder_layers = [
            nn.Linear(backbone_out_size, backbone_out_size),
            nn.ReLU() if cfg.activation == 'relu' else nn.GELU(),
            nn.Linear(backbone_out_size, channel_count)
        ]

        if not cfg.lograte:
            decoder_layers.append(nn.ReLU())
        self.out = nn.Sequential(*decoder_layers)
        self.loss = nn.PoissonNLLLoss(reduction='none', log_input=cfg.lograte)
        self.cfg = cfg.task

    def get_masks(self, loss: torch.Tensor, batch, channel_key=CHANNEL_KEY):
        b, t, a, c = loss.size()
        loss_mask = torch.ones(loss.size(), dtype=torch.bool, device=loss.device)
        if LENGTH_KEY in batch: # only some of b x t are valid
            lengths = batch[LENGTH_KEY] # b of ints < t
            length_mask = torch.arange(t, device=loss.device)[None, :] < lengths[:, None] # B T
            loss_mask = loss_mask & rearrange(length_mask, 'b t -> b t 1 1')
            # import pdb;pdb.set_trace() # TODO needs testing in padding mode
        else:
            length_mask = None
        if channel_key in batch: # only some of b x a x c are valid
            channels = batch[channel_key] # b x a of ints < c
            comparison = repeat(torch.arange(c, device=loss.device), 'c -> 1 a c', a=a)
            channel_mask = comparison < rearrange(channels, 'b a -> b a 1') # B A C
            loss_mask = loss_mask & rearrange(channel_mask, 'b a c -> b 1 a c')
        else:
            channel_mask = None
        return loss_mask, length_mask, channel_mask


    @torch.no_grad()
    def bps(
        self, rates: torch.Tensor, spikes: torch.Tensor, is_lograte=True, mean=True, raw=False,
        length_mask: Optional[torch.Tensor]=None, channel_mask: Optional[torch.Tensor]=None,
        block=False
    ) -> torch.Tensor:
        r""" # tensors B T A C
            Bits per spike, averaged over channels/trials, summed over time.
            Convert extremely uninterpretable NLL into a slightly more interpretable BPS. (0 == constant prediction for BPS)
            For evaluation.
            length_mask: B T
            channel_mask: B A C
        """
        # convenience logic for allowing direct passing of record with additional features
        if is_lograte:
            logrates = rates
        else:
            logrates = (rates + 1e-8).log()
        if spikes.ndim == 5 and logrates.ndim == 4:
            spikes = spikes[..., 0]
        assert spikes.shape == logrates.shape
        nll_model: torch.Tensor = self.loss(logrates, spikes)
        spikes = spikes.float()
        if length_mask is not None:
            nll_model[~length_mask] = 0.
            spikes[~length_mask] = 0
        if channel_mask is not None:
            nll_model[~channel_mask.unsqueeze(1).expand_as(nll_model)] = 0.
            spikes[~channel_mask.unsqueeze(1).expand_as(spikes)] = 0

        nll_model = reduce(nll_model, 'b t a c -> b a c', 'sum')

        if length_mask is not None:
            mean_rates = reduce(spikes, 'b t a c -> b 1 a c', 'sum') / reduce(length_mask, 'b t -> b 1 1 1', 'sum')
        else:
            mean_rates = reduce(spikes, 'b t a c -> b 1 a c')
        if block:
            mean_rates = reduce(mean_rates, 'b 1 a c -> 1 1 a c', 'mean').expand_as(spikes)
        mean_rates = (mean_rates + 1e-8).log()
        nll_null: torch.Tensor = self.loss(mean_rates, spikes)

        if length_mask is not None:
            nll_null[~length_mask] = 0.
        if channel_mask is not None:
            nll_null[~channel_mask.unsqueeze(1).expand_as(nll_null)] = 0.

        nll_null = nll_null.sum(1) # B A C
        # Note, nanmean used to automatically exclude zero firing trials. Invalid items should be reported as nan.s here
        bps_raw: torch.Tensor = ((nll_null - nll_model) / spikes.sum(1) / np.log(2))
        if raw:
            return bps_raw
        bps = bps_raw[(spikes.sum(1) != 0).expand_as(bps_raw)].detach()
        if bps.isnan().any() or bps.mean().isnan().any():
            import pdb;pdb.set_trace() # unnatural - this should only occur if something's really wrong with data
        if mean:
            return bps.mean()
        return bps

class SelfSupervisedInfill(RatePrediction):
    does_update_root = True
    def update_batch(self, batch: Dict[str, torch.Tensor]):
        spikes = batch[DataKey.spikes]
        is_masked = torch.bernoulli(
            torch.full(spikes.size()[:3], self.cfg.mask_ratio, device=spikes.device)
        )
        mask_type = torch.rand_like(is_masked)
        mask_token = mask_type < self.cfg.mask_token_ratio
        mask_random = (mask_type >= self.cfg.mask_token_ratio) & (mask_type < self.cfg.mask_token_ratio + self.cfg.mask_random_ratio)
        is_masked = is_masked.bool()
        mask_token, mask_random = (
            mask_token.bool() & is_masked,
            mask_random.bool() & is_masked,
        )
        target = spikes[..., 0]
        spikes = spikes.clone()

        b, t, a, c, _ = spikes.shape
        if LENGTH_KEY in batch:
            times = rearrange(batch[LENGTH_KEY], 'b -> b 1 1') # 1 = a
        else:
            times = torch.full((b, 1, a), t, device=spikes.device)

        if self.cfg.mask_random_shuffle:
            # How can we generate a random time if we have different bounds? Use a large number and take modulo, roughly fair
            # (note permute doesn't really work if we have ragged times, we risk shuffling in padding)
            random_draw = torch.randint(0, 100000, (b, t, a), device=times.device) % times

            # Use random_draw to index spikes and extract a tensor of size b t a c 1
            time_shuffled_spikes = spikes.gather(1, random_draw.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, c, -1))
            spikes[mask_random] = time_shuffled_spikes[mask_random]
        else:
            # Old implementation
            spikes[mask_random] = torch.randint_like(spikes[mask_random], 0, spikes.max().int().item() + 1)
        spikes[mask_token] = 0 # use zero mask per NDT (Ye 21)
        batch.update({
            DataKey.spikes: spikes,
            'is_masked': is_masked,
            'spike_target': target,
        })
        return batch

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True) -> torch.Tensor:
        rates: torch.Tensor = self.out(backbone_features)
        batch_out = {}
        if Output.logrates in self.cfg.outputs:
            batch_out[Output.logrates] = rates

        if not compute_metrics:
            return batch_out
        spikes = batch['spike_target']
        loss: torch.Tensor = self.loss(rates, spikes)
        # Infill update mask
        loss_mask, length_mask, channel_mask = self.get_masks(loss, batch)
        loss_mask = loss_mask & rearrange(batch['is_masked'], 'b t a -> b t a 1')
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.bps in self.cfg.metrics:
            batch_out[Metric.bps] = self.bps(
                rates, spikes,
                length_mask=length_mask,
                channel_mask=channel_mask
            )

        return batch_out


class NextStepPrediction(TaskPipeline):

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ICMSNextStepPrediction(NextStepPrediction):
    does_update_root = True
    def update_batch(self, batch: Dict[str, torch.Tensor]):
        # Remove final timestep, prepend "initial" quiet recording (for causal)
        spikes = batch[DataKey.spikes]
        spikes = torch.cat([torch.zeros_like(spikes[:,:1]), spikes[:,:-1]], 1)
        batch.update({
            DataKey.spikes: spikes
        })
        return batch

    def get_temporal_context(self, batch: Dict[str, torch.Tensor]):
        parent = super().get_temporal_context(batch)
        return [*parent, batch[DataKey.stim]]


class HeldoutPrediction(RatePrediction):
    r"""
        Regression for co-smoothing
    """
    def __init__(
        self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size * data_attrs.max_arrays,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs,
        )

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True) -> torch.Tensor:
        # torch.autograd.set_detect_anomaly(True)
        backbone_features = rearrange(backbone_features.clone(), 'b t a c -> b t (a c)')
        # backbone_features = rearrange(backbone_features, 'b t a c -> b t (a c)')
        rates: torch.Tensor = self.out(backbone_features)
        batch_out = {}
        if Output.heldout_logrates in self.cfg.outputs:
            batch_out[Output.heldout_logrates] = rates

        if not compute_metrics:
            return batch_out
        spikes = batch[DataKey.heldout_spikes][..., 0]
        loss: torch.Tensor = self.loss(rates, spikes)
        # re-expand array dimension to match API expectation for array dim
        loss = rearrange(loss, 'b t c -> b t 1 c')
        loss_mask, length_mask, channel_mask = self.get_masks(loss, batch, channel_key=HELDOUT_CHANNEL_KEY) # channel_key expected to be no-op since we don't provide this mask
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.co_bps in self.cfg.metrics:
            batch_out[Metric.co_bps] = self.bps(
                rates.unsqueeze(-2), spikes.unsqueeze(-2),
                length_mask=length_mask,
                channel_mask=channel_mask
            )
        if Metric.block_co_bps in self.cfg.metrics:
            batch_out[Metric.block_co_bps] = self.bps(
                rates.unsqueeze(-2), spikes.unsqueeze(-2),
                length_mask=length_mask,
                channel_mask=channel_mask,
                block=True
            )

        return batch_out



# TODO convert to registry
task_modules = {
    ModelTask.infill: SelfSupervisedInfill,
    ModelTask.icms_one_step_ahead: ICMSNextStepPrediction, # ! not implemented, above logic is infill specific
    ModelTask.heldout_decoding: HeldoutPrediction,
}