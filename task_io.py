from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack # baby steps...
from einops.layers.torch import Rearrange
from sklearn.metrics import r2_score
import logging

from config import (
    ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey,
)

from data import DataAttrs, LENGTH_KEY, CHANNEL_KEY, HELDOUT_CHANNEL_KEY
from subjects import subject_array_registry, SortedArrayInfo

from components import SpaceTimeTransformer

# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
SHUFFLE_KEY = "shuffle"

def apply_shuffle(item: torch.Tensor, shuffle: torch.Tensor):
    # item: B T *
    # shuffle: T
    return item.transpose(1, 0)[shuffle].transpose(1, 0)

class TaskPipeline(nn.Module):
    r"""
        Task IO - manages decoder layers, loss functions
        i.e. is responsible for returning loss, decoder outputs, and metrics
    """
    does_update_root = False
    unique_space = False # accept unique space as input?

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ) -> None:
        super().__init__()
        self.pad_value = data_attrs.pad_token
        self.serve_tokens = data_attrs.serve_tokens
        self.serve_tokens_flat = data_attrs.serve_tokens_flat

    def get_masks(self, batch, channel_key=CHANNEL_KEY):
        # loss_mask: b t *
        ref = batch[DataKey.spikes][..., 0]
        if self.serve_tokens_flat:
            b, t, c = ref.size()
            s = 1
        else:
            b, t, s, c = ref.size()
        loss_mask = torch.ones(ref.size(), dtype=torch.bool, device=ref.device)
        if LENGTH_KEY in batch: # only some of b x t are valid
            lengths = batch[LENGTH_KEY] # b of ints < t
            comparison = torch.arange(t, device=ref.device)[None, :]
            if SHUFFLE_KEY in batch:
                comparison = apply_shuffle(comparison, batch[SHUFFLE_KEY])
            length_mask = comparison < lengths[:, None] # B T
            if self.serve_tokens_flat:
                loss_mask = loss_mask & length_mask.unsqueeze(-1)
            else:
                loss_mask = loss_mask & rearrange(length_mask, 'b t -> b t 1 1')
            # import pdb;pdb.set_trace() # TODO needs testing in padding mode
        else:
            length_mask = None
        if channel_key in batch: # only some of b x a x c are valid
            channels = batch[channel_key] # b x a of ints < c (or b x t)
            # Note no shuffling occurs here because 1. channel_key shuffle is done when needed earlier
            # 2. no spatial shuffling occurs so we do need to apply_shuffle(torch.arange(c))
            if self.serve_tokens and not self.serve_tokens_flat: # if there's multiple arrays, this will show up as b x a (separate from spatial dim in tokens)
                # we have B S C, goal is to determine which are real vs padding
                # at our disposal we have the channel count, separated into b x a
                # each a is a number, that was divided by neurons per token to create the spatial distribution
                # to reverse engineer, first compute the relevant spatial tokens for each array
                # neurons_per_token = c
                allocated_space_tokens = torch.ceil(channels / c) # B A
                # I need to somehow distribute the spatial dimension among the arrays
                # there are number of ways to do this; we'll do what seems simplest at this time
                # find any leftover padding, declare zero channels
                comparison = repeat(torch.arange(s, device=ref.device), 's -> 1 s c', c=c)
                pure_padding_tokens = allocated_space_tokens.sum(-1)
                channel_mask = comparison < rearrange(pure_padding_tokens, 'b -> b 1 1')
                # for those fractional tokens, declare the % of channels
                allocated_fractional = channels % c # B A
                comparison = repeat(torch.arange(c, device=ref.device), 'c -> 1 a c', a=channels.size(-1))
                # gather the relevant entries from `chanenl_mask` using `allocated_space_tokens` as an index
                # If we happen to allocate the perfect number, the ceiling will do nothing and our index won't be appropriately "one above" where we want to index
                fractional_index = torch.where(allocated_fractional > 0, allocated_space_tokens - 1, allocated_space_tokens).long()
                channel_mask.scatter_(1,
                    repeat(fractional_index, 'b a -> b a c', c=c),
                    comparison < allocated_fractional.unsqueeze(-1)
                )
            else:
                comparison = repeat(torch.arange(c, device=ref.device), 'c -> 1 s c', s=s)
                channel_mask = comparison < rearrange(channels, 'b t -> b t 1') # dim 2 is either arrays (base case) or tokens (flat)
            if not self.serve_tokens_flat:
                loss_mask = loss_mask & rearrange(channel_mask, 'b a c -> b 1 a c')
            else:
                loss_mask = loss_mask & channel_mask
        else:
            loss_mask = loss_mask[..., 0] # don't specify channel dim if not used, saves HELDOUT case
            channel_mask = None
        return loss_mask, length_mask, channel_mask

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

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
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

    def forward(self, batch, backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        r"""
            By default only return outputs. (Typically used in inference)
            - compute_metrics: also return metrics.
            - eval_mode: Run IO in eval mode (e.g. no masking)
        """
        raise NotImplementedError

class RatePrediction(TaskPipeline):
    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
        decoder: nn.Module | None = None,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.cfg = cfg.task
        if self.serve_tokens_flat:
            assert Metric.bps not in self.cfg.metrics, "bps metric not supported for flat tokens"
        if decoder is not None:
            self.out = decoder
        else:
            readout_size = cfg.neurons_per_token if cfg.transform_space else channel_count
            if getattr(self.cfg, 'unique_no_head', False):
                decoder_layers = []
            elif getattr(self.cfg, 'linear_head', False):
                decoder_layers = [nn.Linear(backbone_out_size, readout_size)]
            else:
                decoder_layers = [
                    nn.Linear(backbone_out_size, backbone_out_size),
                    nn.ReLU() if cfg.activation == 'relu' else nn.GELU(),
                    nn.Linear(backbone_out_size, readout_size)
                ]

            if not cfg.lograte:
                decoder_layers.append(nn.ReLU())

            if cfg.transform_space and not self.serve_tokens: # if serving as tokens, then target has no array dim
                # after projecting, concatenate along the group dimension to get back into channel space
                decoder_layers.append(Rearrange('b t a s_a c -> b t a (s_a c)'))
            self.out = nn.Sequential(*decoder_layers)
        self.loss = nn.PoissonNLLLoss(reduction='none', log_input=cfg.lograte)

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

            block: Whether to get null from full batch (more variable, but std defn)
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
            # spikes[~channel_mask.unsqueeze(1).expand_as(spikes)] = 0 # redundant

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
    unique_space = True
    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        if eval_mode:
            batch.update({
                # don't actually mask
                'is_masked': torch.zeros(spikes.size()[:-2], dtype=torch.bool, device=spikes.device),
                'spike_target': target
            })
            return batch
        is_masked = torch.bernoulli(
            # ! Spatial-masking seems slightly worse on RTT, revisit with tuning + neuron dropout
            torch.full(spikes.size()[:2], self.cfg.mask_ratio, device=spikes.device)
            # torch.full(spikes.size()[:-2], self.cfg.mask_ratio, device=spikes.device)
        ) # B T S or B Token - don't mask part of a token
        if not self.serve_tokens_flat:
            is_masked = is_masked.unsqueeze(-1) # mock spatial masking
        mask_type = torch.rand_like(is_masked)
        mask_token = mask_type < self.cfg.mask_token_ratio
        mask_random = (mask_type >= self.cfg.mask_token_ratio) & (mask_type < self.cfg.mask_token_ratio + self.cfg.mask_random_ratio)
        is_masked = is_masked.bool()
        mask_token, mask_random = (
            mask_token.bool() & is_masked,
            mask_random.bool() & is_masked,
        )

        spikes = spikes.clone()
        if self.cfg.mask_random_shuffle:
            assert not self.serve_tokens, 'shape not updated'
            b, t, a, c, _ = spikes.shape
            if LENGTH_KEY in batch:
                times = rearrange(batch[LENGTH_KEY], 'b -> b 1 1') # 1 = a
            else:
                times = torch.full((b, 1, a), t, device=spikes.device)
            # How can we generate a random time if we have different bounds? Use a large number and take modulo, roughly fair
            # (note permute doesn't really work if we have ragged times, we risk shuffling in padding)
            random_draw = torch.randint(0, 100000, (b, t, a), device=times.device) % times

            # Use random_draw to index spikes and extract a tensor of size b t a c 1
            # TODO update this
            time_shuffled_spikes = spikes.gather(1, random_draw.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, c, -1))
            spikes[mask_random] = time_shuffled_spikes[mask_random]
        else:
            if self.serve_tokens and not self.serve_tokens_flat: # ! Spatial-masking seems slightly worse on RTT, revisit with tuning + neuron dropout
                mask_random = mask_random.expand(-1, -1, spikes.size(2))
                mask_token = mask_token.expand(-1, -1, spikes.size(2))
            spikes[mask_random] = torch.randint_like(spikes[mask_random], 0, spikes[spikes != self.pad_value].max().int().item() + 1)
        spikes[mask_token] = 0 # use zero mask per NDT (Ye 21) # TODO revisit for spatial mode; not important in causal mode
        batch.update({
            DataKey.spikes: spikes,
            'is_masked': is_masked,
            'spike_target': target,
        })
        return batch

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        rates: torch.Tensor = self.out(backbone_features)
        batch_out = {}
        if Output.logrates in self.cfg.outputs:
            assert not self.serve_tokens, 'shape not updated, not too sure what to do here'
            batch_out[Output.logrates] = rates

        if not compute_metrics:
            return batch_out
        spikes = batch['spike_target']
        loss: torch.Tensor = self.loss(rates, spikes)
        # Infill update mask
        loss_mask, length_mask, channel_mask = self.get_masks(batch)
        if Metric.all_loss in self.cfg.metrics:
            batch_out[Metric.all_loss] = loss[loss_mask].mean().detach()
        loss_mask = loss_mask & batch['is_masked'].unsqueeze(-1) # add channel dim
        # loss_mask = loss_mask & rearrange(batch['is_masked'], 'b t s -> b t s 1')
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.bps in self.cfg.metrics:
            batch_out[Metric.bps] = self.bps(
                rates, spikes,
                length_mask=length_mask,
                channel_mask=channel_mask
            )

        return batch_out

class ShuffleInfill(RatePrediction):
    r"""
        Technical design decision note:
        - JY instinctively decided to split up inputs and just carry around split tensors rather than the splitting metadata.
        - This is somewhat useful in the end (rather than the unshuffling solution) as we can simply collect the masked crop
        - However the code is pretty dirty and this may eventually change

    """
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
        self.cfg = cfg.task
        assert self.serve_tokens and self.serve_tokens_flat, 'other paths not implemented'
        max_spatial_tokens = round(
            data_attrs.max_channel_count / cfg.neurons_per_token
        )
        assert cfg.encode_decode, 'non-symmetric evaluation not implemented (since this task crops)'
        # ! Need to figure out how to wire different parameters e.g. num layers here
        self.decoder = SpaceTimeTransformer(
            cfg.transformer,
            max_spatial_tokens=max_spatial_tokens,
            n_layers=cfg.decoder_layers,
        )
        out_layers = [
            nn.Linear(backbone_out_size, cfg.neurons_per_token)
        ]
        if not cfg.lograte:
            out_layers.append(nn.ReLU())
        self.out = nn.Sequential(*out_layers)
        self.loss = nn.PoissonNLLLoss(reduction='none', log_input=cfg.lograte)
        self.mask_token = nn.Parameter(torch.randn(cfg.hidden_size))

    does_update_root = True
    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        r"""
            Shuffle inputs, keep only what we need for evaluation
        """
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        if eval_mode:
            batch.update({
                SHUFFLE_KEY: torch.arange(spikes.size(1), device=spikes.device),
                'spike_target': target
            })
            return batch
        # spikes: B T S H or B T H (no array support)
        # TODO (low-pri) also support spacetime shuffle
        shuffle = torch.randperm(spikes.size(1), device=spikes.device)
        encoder_frac = int((1 - self.cfg.mask_ratio) * spikes.size(1))
        # shuffle_spikes = spikes.gather(1, shuffle.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spikes.size(2), spikes.size(3)))
        for key in [DataKey.time, DataKey.position, CHANNEL_KEY]:
            if key in batch:
                shuffled = apply_shuffle(batch[key], shuffle)
                batch.update({
                    key: shuffled[:, :encoder_frac],
                    f'{key}_target': shuffled[:, encoder_frac:],
                })
        # import pdb;pdb.set_trace()
        batch.update({
            DataKey.spikes: apply_shuffle(spikes, shuffle)[:,:encoder_frac],
            'spike_target': apply_shuffle(target, shuffle)[:,encoder_frac:],
            'encoder_frac': encoder_frac,
            SHUFFLE_KEY: shuffle, # seems good to keep around...
        })
        return batch

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        # B T
        target = batch['spike_target']
        if target.ndim == 5:
            raise NotImplementedError("cannot even remember what this should look like")
            # decoder_mask_tokens = repeat(self.mask_token, 'h -> b t s h', b=target.size(0), t=target.size(1), s=target.size(2))
        else:
            decoder_mask_tokens = repeat(self.mask_token, 'h -> b t h', b=target.size(0), t=target.size(1))
            decoder_input = torch.cat([backbone_features, decoder_mask_tokens], dim=1)
            times = torch.cat([batch[DataKey.time], batch[f'{DataKey.time}_target']], 1)
            positions = torch.cat([batch[DataKey.position], batch[f'{DataKey.position}_target']], 1)

            trial_context = []
            for key in ['session', 'subject', 'task']:
                if batch.get(key, None) is not None:
                    trial_context.append(batch[key])

            if LENGTH_KEY in batch:
                token_position = rearrange(batch[SHUFFLE_KEY], 't -> () t')
                temporal_padding_mask = token_position >= rearrange(batch[LENGTH_KEY], 'b -> b ()')
            reps: torch.Tensor = self.decoder(
                decoder_input,
                trial_context=trial_context,
                temporal_padding_mask=temporal_padding_mask,
                space_padding_mask=None, # TODO implement (low pri)
                causal=False, # TODO implement (low pri)
                times=times,
                positions=positions
            )
            reps = reps[:, -target.size(1):]
            rates = self.out(reps)
        batch_out = {}
        assert not Metric.bps in self.cfg.metrics, 'not supported'
        assert not Output.logrates in self.cfg.outputs, 'not implemented'

        if not compute_metrics:
            return batch_out
        loss: torch.Tensor = self.loss(rates, target) # b t' c
        # get_masks
        loss_mask = torch.ones(loss.size(), device=loss.device, dtype=torch.bool)
        # note LENGTH_KEY and CHANNEL_KEY are for padding tracking
        # while DataKey.time and DataKey.position are for content
        if LENGTH_KEY in batch:
            token_position = rearrange(batch[SHUFFLE_KEY][batch['encoder_frac']:], 't -> () t')
            length_mask = token_position < rearrange(batch[LENGTH_KEY], 'b -> b ()')
            loss_mask = loss_mask & length_mask.unsqueeze(-1)
        if CHANNEL_KEY in batch:
            # CHANNEL_KEY padding tracking has already been shuffled
            # And within each token, we just have c channels to track, always in order
            comparison = repeat(torch.arange(loss.size(-1), device=loss.device), 'c -> 1 t c', t=loss.size(1)) # ! assuming flat - otherwise we need the space dimension as well.
            channel_mask = comparison < batch[f'{CHANNEL_KEY}_target'].unsqueeze(-1) # unsqueeze the channel dimension
            loss_mask = loss_mask & channel_mask
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        return batch_out


# todo make shuffle next step prediction
class NextStepPrediction(RatePrediction):
    r"""
        One-step-ahead modeling prediction.
        Note while pretraining necesarily should be causal (no way of preventing ctx bleed across layers)
        We can still use a semi-causal decoder (however much context we can afford).
    """
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        backbone_features: torch.Tensor,
        compute_metrics=True,
        eval_mode=False,
    ) -> torch.Tensor:
        rates: torch.Tensor = self.out(backbone_features)
        # Match API of infill; just roll, don't use first timestep
        rates = torch.roll(rates, 1, dims=1)

        batch_out = {}
        if Output.logrates in self.cfg.outputs:
            batch_out[Output.logrates] = rates

        if not compute_metrics:
            return batch_out
        loss: torch.Tensor = self.loss(rates, batch[DataKey.spikes][...,0])
        loss_mask, length_mask, channel_mask = self.get_masks(batch)
        loss_mask[:, 0] = False # don't compute loss on first timestep
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.bps in self.cfg.metrics:
            batch_out[Metric.bps] = self.bps(
                rates[:,1:], batch[DataKey.spikes][:,1:],
                length_mask=length_mask[:,1:],
                channel_mask=channel_mask
            )

        return batch_out

class ICMSNextStepPrediction(NextStepPrediction):
    does_update_root = True
    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        raise NotImplementedError # ! really unclear whether we want to update spikes like this...
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
        self.concatenating = cfg.transform_space and not data_attrs.serve_tokens
        if self.concatenating:
            # It takes too much memory to concatenate full length tokens x however many tokens there are
            # Project down to neuron dim first.
            decoder_layers = [
                nn.Linear(backbone_out_size, backbone_out_size),
                nn.ReLU() if cfg.activation == 'relu' else nn.GELU(),
                nn.Linear(backbone_out_size, cfg.neurons_per_token),
                Rearrange('b t a s_a c -> b t (a s_a c)'),
                nn.ReLU() if cfg.activation == 'relu' else nn.GELU(),
                nn.Linear(data_attrs.max_arrays * data_attrs.max_channel_count, channel_count),
            ]
            if not cfg.lograte:
                decoder_layers.append(nn.ReLU())
            decoder = nn.Sequential(*decoder_layers)
        else:
            backbone_out_size = backbone_out_size * data_attrs.max_arrays
            decoder = None
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs,
            decoder=decoder
        )

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        if not self.concatenating:
            backbone_features = rearrange(backbone_features.clone(), 'b t a c -> b t (a c)')
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
        loss_mask, length_mask, channel_mask = self.get_masks(batch, channel_key=HELDOUT_CHANNEL_KEY) # channel_key expected to be no-op since we don't provide this mask
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

class BehaviorRegression(TaskPipeline):
    def __init__(
        self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.cfg = cfg.task
        r"""
            Because this is not intended to be a joint task, we will not make the decoder deep (expectation is tuning)
        """
        # For linear decoder, deal with multiple arrays by concatenating
        if cfg.transform_space and not data_attrs.serve_tokens:
            self.decoder = nn.Sequential(
                Rearrange('b t a s h -> b t (a s h)'),
                nn.Linear(backbone_out_size * round(data_attrs.max_channel_count / data_attrs.neurons_per_token) * data_attrs.max_arrays, data_attrs.behavior_dim)
            )
        else:
            self.decoder = nn.Sequential(
                Rearrange('b t a c -> b t (a c)'),
                nn.Linear(backbone_out_size * data_attrs.max_arrays, data_attrs.behavior_dim)
            )
        self.bhvr_lag_bins = round(self.cfg.behavior_lag / data_attrs.bin_size_ms)

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        batch_out = {}
        # backbone_features = rearrange(backbone_features, 'b t a c -> b t (a c)')
        bhvr = self.decoder(backbone_features)
        if self.bhvr_lag_bins:
            bhvr = bhvr[:, :-self.bhvr_lag_bins]
            bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)
        if Output.behavior_pred in self.cfg.outputs:
            batch_out[Output.behavior_pred] = bhvr
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = batch[self.cfg.behavior_target]
        if not compute_metrics:
            return batch_out
        # Compute loss
        bhvr_tgt = batch[self.cfg.behavior_target]
        loss = F.mse_loss(bhvr, bhvr_tgt, reduction='none')
        _, length_mask, _ = self.get_masks(batch, channel_key=None)

        length_mask[:, :self.bhvr_lag_bins] = False # don't compute loss for lagged out timesteps
        batch_out['loss'] = loss[length_mask].mean()
        if Metric.kinematic_r2 in self.cfg.metrics:
            valid_bhvr = bhvr[length_mask]
            valid_tgt = bhvr_tgt[length_mask]
            batch_out[Metric.kinematic_r2] = r2_score(valid_bhvr.detach().cpu(), valid_tgt.detach().cpu(), multioutput='raw_values')
            # print(f'true: {bhvr_tgt[length_mask][:10,0]}')
            # print(f'pred: {bhvr[length_mask][:10,0]}')
            # print(f'loss: {loss[length_mask][:10,0]}')
            # print(f'x_r2: {batch_out[Metric.kinematic_r2][0]}')
        return batch_out

task_modules = {
    ModelTask.infill: SelfSupervisedInfill,
    ModelTask.shuffle_infill: ShuffleInfill,
    ModelTask.next_step_prediction: NextStepPrediction, # ! not implemented, above logic is infill specific
    ModelTask.heldout_decoding: HeldoutPrediction,
    ModelTask.kinematic_decoding: BehaviorRegression,
}