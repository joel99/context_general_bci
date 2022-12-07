from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack # baby steps...

import logging

from config import (
    ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
)

from data import DataAttrs, LENGTH_KEY, CHANNEL_KEY
from subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from backbones import TemporalTransformer

class BrainBertInterface(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(self, cfg: ModelConfig, data_attrs: DataAttrs):
        super().__init__() # store cfg
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.backbone = TemporalTransformer(self.cfg.transformer)
        self.data_attrs = None
        self.bind_io(data_attrs)

    def bind_io(self, data_attrs: DataAttrs):
        r"""
            Add context-specific input/output parameters.

            Ideally, we will just bind embedding layers here, but there may be some MLPs.
        """
        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            assert data_attrs.context.session, "Session embedding strategy requires session in data"
            if len(data_attrs.context.session) == 1:
                logging.warn('Using session embedding strategy with only one session. This is probably a mistake.')
        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            assert data_attrs.context.subject, "Subject embedding strategy requires subject in data"
            if len(data_attrs.context.subject) == 1:
                logging.warn('Using subject embedding strategy with only one subject. This is probably a mistake.')
        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            assert data_attrs.context.array, "Array embedding strategy requires array in data"
            if len(data_attrs.context.array) == 1:
                logging.warn('Using array embedding strategy with only one array. This is probably a mistake.')
        if self.data_attrs is not None: # IO already exists
            import pdb;pdb.set_trace() # check this named_children call
            # TODO update this to re-assign any preserved io
            for n, m in self.named_children(): # clean old io
                if n == 'backbone':
                    continue
                del m
        self.data_attrs = data_attrs

        # Guardrails (to remove)
        assert len(self.data_attrs.context.task) <= 1, "Only tested for single task"
        assert self.cfg.array_embed_strategy == EmbedStrat.none, "Array embed strategy not yet implemented"
        assert len(data_attrs.context.subject) <= 1, "Only tested for single subject"


        project_size = self.cfg.hidden_size
        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            self.session_embed = nn.Embedding(len(self.data_attrs.context.session), self.cfg.session_embed_size)
            if self.cfg.session_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.session_embed_size
            elif self.cfg.session_embed_strategy == EmbedStrat.token:
                assert self.cfg.session_embed_size == self.cfg.hidden_size
                self.session_flag = nn.Parameter(torch.zeros(self.cfg.session_embed_size))

        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            self.subject_embed = nn.Embedding(len(self.data_attrs.context.subject), self.cfg.subject_embed_size)
            if self.cfg.subject_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.subject_embed_size
            elif self.cfg.subject_embed_strategy == EmbedStrat.token:
                assert self.cfg.subject_embed_size == self.cfg.hidden_size
                self.subject_flag = nn.Parameter(torch.zeros(self.cfg.subject_embed_size))

        # TODO handle array embed padding
        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            self.array_embed = nn.Embedding(len(self.data_attrs.context.array) + 1, self.cfg.array_embed_size)
            # # +1 is for padding (i.e. self.array_embed[-1] = padding)
            if self.cfg.array_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.subject_embed_size
            elif self.cfg.array_embed_strategy == EmbedStrat.token:
                assert self.cfg.array_embed_size == self.cfg.hidden_size
                self.array_flag = nn.Parameter(torch.zeros(self.cfg.array_embed_size))

        if project_size is not self.cfg.hidden_size:
            self.context_project = nn.Sequential(
                nn.Linear(project_size, self.cfg.hidden_size),
                nn.ReLU()
            )
        else:
            self.context_project = None

        if self.cfg.task.task in [ModelTask.icms_one_step_ahead, ModelTask.infill]:
            # bookmark: multi-array readin should be done here
            # Note in general the data module will be responsible for providing array masks
            if self.data_attrs.max_channel_count > 0: # there is padding
                channel_count = self.data_attrs.max_channel_count
            else:
                # * Just project all channels.
                # Doesn't (yet) support separate array projections.
                # Doesn't (yet) support task-subject specific readin.
                # ? I am unclear how Talukder managed to have mixed batch training if different data was shaped different sizes.
                # * Because we only ever train on one subject in this strategy, all registered arrays must belong to that subject.
                # * A rework will be needed if we want to do this lookup grouped per subject
                assert self.cfg.readin_strategy == EmbedStrat.project, 'Ragged array readin only implemented for project readin strategy'
                assert len(data_attrs.context.subject) <= 1, "Only implemented for single subject (likely need padding for mixed batches)"

                for a in self.data_attrs.context.array:
                    assert not isinstance(subject_array_registry.query_by_array(a), SortedArrayInfo), "actual mixed readins per session not yet implemented"
                channel_count = sum(
                    subject_array_registry.query_by_array(a).get_channel_count() for a in self.data_attrs.context.array
                ) * self.data_attrs.spike_dim
            self.readin = nn.Linear(channel_count, self.cfg.hidden_size)

            # TODO add readin for the stim array (similar attr)
            if self.cfg.task.task == ModelTask.icms_one_step_ahead:
                raise NotImplementedError

            decoder_layers = [
                nn.Linear(self.backbone.out_size, channel_count)
            ]

            if not self.cfg.lograte:
                decoder_layers.append(nn.ReLU())
            self.out = nn.Sequential(*decoder_layers)
            self.loss = nn.PoissonNLLLoss(reduction='none', log_input=self.cfg.lograte)


    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
            Format spikes and context into tokens for backbone.
            In:
                spikes: B T A C H=1 (features provided on channel dim for principles but functionally useless)
            Returns:
                state_in: B x T x A x H (A should be flattened in backbone)
                static_context: List(T') [B x H]
                temporal_context: List(?) [B x T x H]
        """
        spikes = rearrange(batch[DataKey.spikes], 'b t a c h -> b t a (c h)')
        if self.cfg.task.task == ModelTask.icms_one_step_ahead:
            # Remove final timestep, prepend "initial" quiet recording
            state_in = torch.cat([torch.zeros_like(spikes[:,:1]), spikes[:,:-1]], 1)
            temporal_context = batch[DataKey.stim]
        else:
            state_in = spikes
            temporal_context = []
        state_in = self.readin(state_in) # b t a h

        static_context = []
        project_context = [] # only for static info
        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            session: torch.Tensor = self.session_embed(batch[MetaKey.session]) # B x H
            if self.cfg.session_embed_strategy == EmbedStrat.token:
                session = session + self.session_flag # B x H
                static_context.append(session)
            elif self.cfg.session_embed_strategy == EmbedStrat.concat:
                session = repeat(session, 'b h -> b t h', t=state_in.shape[1])
                project_context.append(session)

        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            subject: torch.Tensor = self.subject_embed(batch[MetaKey.subject]) # B x H
            if self.cfg.subject_embed_strategy == EmbedStrat.token:
                subject = subject + self.subject_flag
                static_context.append(subject)
            elif self.cfg.subject_embed_strategy == EmbedStrat.concat:
                subject = repeat(subject, 'b h -> b t h', t=state_in.shape[1])
                project_context.append(subject)

        assert self.cfg.array_embed_strategy is EmbedStrat.none, "Not implemented"

        # TODO support temporal embed + temporal project
        # Do not concat static context - list default is easier to deal with
        # static_context = rearrange(static_context, 't0 b h -> b t0 h') if static_context else None
        if project_context: # someone wanted it
            # B T' H, and we want to merge into B T A H (specifically add T' to each token)
            augmented_tokens, ps = pack([state_in, *project_context], 'b * h')
            augmented_tokens = self.context_project(augmented_tokens)
            state_in = rearrange(augmented_tokens, ps, 'b (t a) h', t=state_in.size(1))
        return state_in, static_context, temporal_context

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_in, trial_context, temporal_context = self._prepare_inputs(batch)
        if LENGTH_KEY in batch:
            temporal_padding_mask = torch.arange(state_in.size(1), device=state_in.device)[None, :] >= batch[LENGTH_KEY][:, None] # -> B T
        else:
            temporal_padding_mask = None

        # Note that fine-grained channel mask doesn't matter in forward (sub-token padding is handled in loss calculation externally)
        # But we do want to exclude fully-padded arrays from computation
        array_padding_mask = batch[CHANNEL_KEY] == 0  if CHANNEL_KEY in batch else None # b x a of ints < c

        outputs: torch.Tensor = self.backbone(
            state_in,
            trial_context=trial_context,
            temporal_context=temporal_context,
            temporal_padding_mask=temporal_padding_mask,
            array_padding_mask=array_padding_mask,
            causal=self.cfg.task.task == ModelTask.icms_one_step_ahead,
        ) # B x T x A x H # TODO satisfy shape
        if outputs.isnan().any(): # I have no idea why, but something in s61 or s62 throws a nan
            # And strangely, I can't repro by invoking forward again.
            # Leaving for now to see if it happens after padding refactor
            import pdb;pdb.set_trace()

        if self.cfg.task.task in [ModelTask.icms_one_step_ahead, ModelTask.infill]:
            rates = self.out(outputs)
            return {
                Output.rates: rates,
            }


    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
            batch provided contains all configured data_keys and meta_keys

            TODO:
            - Fix: targets are keyed/id-ed per task; there is just a single target variable we're hoping is right
            - ?: Ideally the payloads could be mroe strongly typed.

            Example shapes:
                spikes: B T A C H=1 (C is electrode channel)
                stim: B T C H
                channel_counts: B A (counts per array)
        """
        target = None
        # TODO figure out how to wrap this ICMS code in a task abstraction
        if self.cfg.task.task in [ModelTask.icms_one_step_ahead, ModelTask.infill]:
            spikes = batch[DataKey.spikes]

        if self.cfg.task.task == ModelTask.icms_one_step_ahead:
            target = spikes[..., 0]
            # ! dont' forget a cast to float
        elif self.cfg.task.task == ModelTask.infill:
            is_masked = torch.bernoulli(
                torch.full(spikes.size()[:3], self.cfg.task.mask_ratio, device=spikes.device)
            )
            mask_token = torch.bernoulli(torch.full_like(is_masked, self.cfg.task.mask_token_ratio))
            mask_random = torch.bernoulli(torch.full_like(is_masked, self.cfg.task.mask_random_ratio))
            is_masked = is_masked.bool()
            mask_token, mask_random = (
                mask_token.bool() & is_masked,
                mask_random.bool() & is_masked,
            )
            target = spikes[..., 0]
            spikes = spikes.clone()
            spikes[mask_random] = torch.randint_like(spikes[mask_random], 0, spikes.max().int().item() + 1)
            spikes[mask_token] = 0 # use zero mask per NDT (Ye 21)
            batch = {
                **batch,
                # DataKey.spikes: spikes,
                DataKey.spikes: torch.as_tensor(spikes, dtype=torch.float),
            }


        predict = self(batch) # B T A C
        if self.cfg.task.task in [ModelTask.icms_one_step_ahead, ModelTask.infill]:
            loss: torch.Tensor = self.loss(predict[Output.rates], target)

        b, t, a, c = loss.size()
        loss_mask = torch.ones(loss.size(), dtype=torch.bool, device=loss.device)
        if LENGTH_KEY in batch: # only some of b x t are valid
            lengths = batch[LENGTH_KEY] # b of ints < t
            length_mask = torch.arange(t, device=spikes.device)[None, :] < lengths[:, None] # B T
            loss_mask = loss_mask & rearrange(length_mask, 'b t -> b t 1 1')
            # import pdb;pdb.set_trace() # TODO needs testing in padding mode
        if CHANNEL_KEY in batch: # only some of b x a x c are valid
            channels = batch[CHANNEL_KEY] # b x a of ints < c
            comparison = repeat(torch.arange(c, device=spikes.device), 'c -> 1 a c', a=a)
            channel_mask = comparison < rearrange(channels, 'b a -> b a 1') # B A C
            loss_mask = loss_mask & rearrange(channel_mask, 'b a c -> b 1 a c')
        else:
            channel_mask = None
        if self.cfg.task.task == ModelTask.infill:
            loss_mask = loss_mask & rearrange(is_masked, 'b t a -> b t a 1')
        loss = loss[loss_mask]
        batch_out = {'loss': loss.mean()}
        if Metric.bps in self.cfg.task.metrics:
            batch_out[Metric.bps] = self.bps(
                predict[Output.rates],
                target,
                length_mask=length_mask,
                channel_mask=channel_mask
            )
        return batch_out

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        return self(x)

    # =================== Interface IO ===================
    def load_from_checkpoint(self, checkpoint_path: Path | str, data_attrs: DataAttrs, **kwargs):
        r"""
            Load backbone, and determine which parts of the rest of the model to load based on `data_attrs`
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt_data_attrs = ckpt['data_attrs']
        super().load_from_checkpoint(checkpoint_path, ckpt_data_attrs, **kwargs)
        self.bind_io(data_attrs)
        import pdb;pdb.set_trace()
        raise NotImplementedError # Not tested, look through

    # ==================== Utilities ====================
    def transform_rates(
        self,
        logrates: List[torch.Tensor] | torch.Tensor,
        exp=True,
        normalize_hz=False
    ) -> torch.Tensor:
        r"""
            Convenience wrapper for analysis.
            logrates: Raw model output from forward pass. Can be list of batches predictions.
            exp: Should exponentiate?
            normalize_hz: Should normalize to spikes per second (instead of spikes per bin)?
        """
        out = logrates
        if isinstance(out, list):
            out = torch.cat(out)
        if exp:
            out = out.exp()
        if normalize_hz:
            out = out / self.data_attrs.bin_size_ms
        return out

    def bps(
        self, rates: torch.Tensor, spikes: torch.Tensor, is_lograte=True, mean=True, raw=False,
        length_mask: Optional[torch.Tensor]=None, channel_mask: Optional[torch.Tensor]=None
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
            mean_rates = (spikes.sum(1) / length_mask.float().sum(1, keepdim=True)).unsqueeze(1) # B A C / B -> B T A C
        else:
            mean_rates = spikes.mean(1, keepdim=True)
        mean_rates = (mean_rates + 1e-8).log()
        nll_null: torch.Tensor = self.loss(mean_rates, spikes) # B T A C

        if length_mask is not None:
            nll_null[~length_mask] = 0.
        if channel_mask is not None:
            nll_null[~channel_mask.unsqueeze(1).expand_as(nll_null)] = 0.

        nll_null = nll_null.sum(1) # B A C
        # Note, nanmean used to automatically exclude zero firing trials. Invalid items should be reported as nan.s here
        # TODO confirm
        bps_raw: torch.Tensor = ((nll_null - nll_model) / spikes.sum(1) / np.log(2))
        if raw:
            return bps_raw
        bps = bps_raw[(spikes.sum(1) != 0).expand_as(bps_raw)].detach()
        if mean:
            return bps.mean()
        return bps

    # ==================== Optimization ====================
    def predict_step(
        self, batch
    ):
        return self.predict(batch)

    def training_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.log('train_loss', metrics['loss'])
        for m in self.cfg.task.metrics:
            self.log(f'train_{m}', metrics[m])
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.log('val_loss', metrics['loss'])
        for m in self.cfg.task.metrics:
            self.log(f'val_{m}', metrics[m])
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.log('test_loss', metrics['loss'])
        for m in self.cfg.task.metrics:
            self.log(f'test_{m}', metrics[m])
        return metrics['loss']

    def configure_optimizers(self):
        return {
            'optimizer': optim.AdamW(
                self.parameters(),
                lr=self.cfg.lr_init,
                weight_decay=self.cfg.weight_decay
            ),
            'monitor': 'val_loss'
        }