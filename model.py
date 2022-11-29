from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl
from config import (
    ModelConfig, Task, Metric, Output, LENGTH, EmbedStrat, DataKey, MetaKey
)

from data import DataAttrs
from array_locations import subject_to_array
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

        self.backbone = TemporalTransformer(
            self.cfg.hidden_size, self.cfg.transformer
        )
        self.data_attrs = None
        self.bind_io(data_attrs)

        # TODO add some check to compare whether dataset provides everything model needs for tasks
        # (possibly should be in run script)

    def bind_io(self, data_attrs: DataAttrs):
        if self.data_attrs is not None: # IO already exists
            import pdb;pdb.set_trace() # check this named_children call
            # TODO update this to re-assign any preserved io
            for n, m in self.named_children(): # clean old io
                if n == 'backbone':
                    continue
                del m
        self.data_attrs = data_attrs
        assert len(self.data_attrs.context.task) == 1, "Only implemented for single task"
        assert self.cfg.array_embed_strategy == ""
        assert len(data_attrs.context.subject) == 1, "Only implemented for single subject (likely need padding for mixed batches)"

        project_size = self.cfg.hidden_size
        if self.cfg.session_embed_strategy == EmbedStrat.concat:
            self.session_embed = nn.Embedding(len(self.data_attrs.context.session), self.cfg.session_embed_size)
            project_size += self.cfg.session_embed_size
        elif self.cfg.session_embed_strategy == EmbedStrat.token:
            assert self.cfg.session_embed_size == self.cfg.hidden_size
            self.session_embed = nn.Embedding(len(self.data_attrs.context.session), self.cfg.session_embed_size)
            self.session_flag = nn.Parameter(torch.zeros(self.cfg.session_embed_size))

        if self.cfg.subject_embed_strategy == EmbedStrat.concat:
            self.subject_embed = nn.Embedding(len(self.data_attrs.context.subject), self.cfg.subject_embed_size)
            project_size += self.cfg.subject_embed_size
        elif self.cfg.subject_embed_strategy == EmbedStrat.token:
            assert self.cfg.subject_embed_size == self.cfg.hidden_size
            self.subject_embed = nn.Embedding(len(self.data_attrs.context.subject), self.cfg.subject_embed_size)
            self.subject_flag = nn.Parameter(torch.zeros(self.cfg.subject_embed_size))

        if project_size is not self.cfg.hidden_size:
            self.context_project = nn.Sequential(
                nn.Linear(project_size, self.cfg.hidden_size),
                nn.ReLU()
            )
        else:
            self.context_project = None

        if self.cfg.task.task in [Task.icms_one_step_ahead, Task.infill]:
            # bookmark: multi-array readin should be done here
            assert len(data_attrs.context.subject) == 1, "Only implemented for single subject (likely need padding for mixed batches)"
            # readin = []
            # for subject in self.data_attrs.context.subject:
            #     channel_count = subject_to_array[subject].channel_count
            #     readin.append(nn.Linear(channel_count, self.cfg.hidden_size))
            #     # for array in self.data_attrs.context.array: # TODO array subselection
            # self.readin = nn.ModuleList(readin)
            subject = self.data_attrs.context.subject[0]
            channel_count = subject_to_array[subject].channel_count
            self.readin = nn.Linear(channel_count, self.cfg.hidden_size)

            # TODO add something for the stim array (similar attr)
            if self.cfg.task.task == Task.icms_one_step_ahead:
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
            Format spikes and context tokens for backbone. (output T B H)
            Returns:
                state_in: T x B x H
                static_context: T' x B x H
                temporal_context: T x B x H
        """
        if self.cfg.task.task == Task.icms_one_step_ahead:
            spikes = batch[DataKey.spikes]
            # Remove final timestep, prepend "initial" quiet recording
            state_in = torch.cat([torch.zeros_like(spikes[:,:1]), spikes[:,:-1]], 1)
            temporal_context = batch[DataKey.stim]
        else:
            assert False, "Only implemented for ICMS"

        static_context = []
        project_context = []
        if self.cfg.session_embed_strategy == EmbedStrat.token:
            session: torch.Tensor = self.session_embed(batch[MetaKey.session])
            session = session + self.session_flag
            static_context.append(session.unsqueeze(0))
        elif self.cfg.session_embed_strategy == EmbedStrat.concat:
            session = self.session_embed(batch[MetaKey.session]) # B H
            session = session.unsqueeze(0).repeat(state_in.shape[0], 1, 1) # T B H
            project_context.append(session)

        if self.cfg.subject_embed_strategy == EmbedStrat.token:
            subject: torch.Tensor = self.subject_embed(batch[MetaKey.subject])
            subject = subject + self.subject_flag
            static_context.append(subject.unsqueeze(0))
        elif self.cfg.subject_embed_strategy == EmbedStrat.concat:
            subject = self.subject_embed(batch[MetaKey.subject])
            subject = subject.unsqueeze(0).repeat(state_in.shape[0], 1, 1)
            project_context.append(subject)

        static_context = torch.cat(static_context) if static_context else None
        if project_context: # someone wanted it
            state_in = self.context_project(torch.cat([state_in, *project_context], 2))
        return state_in, static_context, temporal_context

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_in, trial_context, temporal_context = self._prepare_inputs(batch)
        if LENGTH in batch:
            padding_mask = torch.arange(state_in.size(0), device=state_in.device)[None, :] >= batch[LENGTH][:, None] # -> B T
        else:
            padding_mask = None
        outputs = self.backbone(
            state_in,
            trial_context=trial_context,
            temporal_context=temporal_context,
            padding_mask=padding_mask
        ) # T x B x H
        if outputs.isnan().any(): # I have no idea why, but something in s61 or s62 throws a nan
            # And strangely, I can't repro by invoking forward again.
            # Leaving for now to see if it happens after padding refactor
            import pdb;pdb.set_trace()

        if self.cfg.task.task in [Task.icms_one_step_ahead, Task.infill]:
            rates = self.out(outputs)
            rates = rates.permute(1, 0, 2) # B T C
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
                spikes: B T C H=1 (C is electrode channel)
                stim: B T C H
        """
        # TODO figure out how to wrap this ICMS code in a task abstraction
        if self.cfg.task.task in [Task.icms_one_step_ahead, Task.infill]:
            spikes = batch[DataKey.spikes]
            target = spikes[..., 0]

        if self.cfg.task.task == Task.icms_one_step_ahead:
            pass
        elif self.cfg.task.task == Task.infill:
            is_masked = torch.bernoulli(
                torch.full((spikes.size(0), spikes.size(1)), self.cfg.task.mask_ratio, device=spikes.device)
            )
            mask_token = torch.bernoulli(torch.full_like(is_masked, self.cfg.task.mask_token_ratio))
            mask_random = torch.bernoulli(torch.full_like(is_masked, self.cfg.task.mask_random_ratio))
            is_masked = is_masked.bool()
            mask_token, mask_random = (
                mask_token.bool() & is_masked,
                mask_random.bool() & is_masked,
            )
            spikes = spikes.clone()
            spikes[mask_random] = torch.randint_like(spikes[mask_random], 0, spikes.max() + 1)
            spikes[mask_token] = 0 # use zero mask per NDT (Ye 21)

        predict = self(batch) # B T C
        if self.cfg.task.task in [Task.icms_one_step_ahead, Task.infill]:
            loss = self.loss(predict[Output.rates], target)

        loss_mask = torch.ones((loss.size(0), loss.size(1)), dtype=torch.bool, device=loss.device)
        if LENGTH in batch:
            lengths = batch[LENGTH]
            length_mask = torch.arange(spikes.size(1), device=spikes.device)[None, :] < lengths[:, None] # B T
            loss_mask = loss_mask & length_mask
        if self.cfg.task.task == Task.infill:
            loss_mask = loss_mask & is_masked
        loss = loss[loss_mask]
        batch_out = {'loss': loss.mean()}
        if Metric.bps in self.cfg.task.metrics:
            batch_out[Metric.bps] = self.bps(predict[Output.rates], target, length_mask=length_mask)
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
        length_mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        r""" # tensors B T C
            Bits per spike, averaged over channels/trials, summed over time.
            Convert extremely uninterpretable NLL into a slightly more interpretable BPS. (0 == constant prediction for BPS)
            For evaluation.
            length_mask: multisession, B T
        """
        # convenience logic for allowing direct passing of record with additional features
        if is_lograte:
            logrates = rates
        else:
            logrates = (rates + 1e-8).log()
        if spikes.ndim == 4 and logrates.ndim == 3:
            spikes = spikes[..., 0]
        assert spikes.shape == logrates.shape
        nll_model: torch.Tensor = self.loss(logrates, spikes)
        nll_model[~length_mask] = 0.
        nll_model = nll_model.sum(1)
        # import pdb;pdb.set_trace()
        if length_mask is not None:
            # take sum and divide by length_mask
            spikes = spikes.clone() # due to in place op
            spikes[~length_mask] = 0
            mean_rates = (spikes.float().sum(1) / length_mask.float().sum(1, keepdim=True)).unsqueeze(1) # B C / B
        else:
            mean_rates = spikes.float().mean(1, keepdim=True)
        mean_rates = (mean_rates + 1e-8).log()
        nll_null: torch.Tensor = self.loss(mean_rates, spikes)
        if length_mask is not None:
            nll_null[~length_mask] = 0.
        nll_null = nll_null.sum(1) # 1 1 C -> B C
        # Note, nanmean used to automatically exclude zero firing trials
        bps_raw: torch.Tensor = ((nll_null - nll_model) / spikes.sum(1) / np.log(2))
        if raw:
            return bps_raw
        bps = bps_raw[spikes.sum(1) != 0].detach()
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