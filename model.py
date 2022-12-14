from typing import Tuple, Dict, List, Optional, Any
import dataclasses
import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack, unpack # baby steps...

import logging

from config import (
    ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey, TaskConfig
)

from data import DataAttrs, LENGTH_KEY, CHANNEL_KEY
from subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from backbones import TemporalTransformer
from task_io import task_modules

logger = logging.getLogger(__name__)

class BrainBertInterface(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(self, cfg: ModelConfig, data_attrs: DataAttrs):
        super().__init__() # store cfg
        self.save_hyperparameters()
        self.cfg = cfg
        self.backbone = TemporalTransformer(self.cfg.transformer)
        self.data_attrs = data_attrs
        self.bind_io()

    def diff_cfg(self, cfg: ModelConfig):
        r"""
            Check if cfg is different from current cfg (used when initing)
        """
        self_copy = self.cfg.copy()
        # Things that are allowed to change on init
        for safe_attr in [
            'task',
            'weight_decay',
            'dropout',
            'lr_init',
            'lr_schedule',
            'lr_ramp_steps',
            'lr_ramp_init_factor',
            'lr_decay_steps',
            'lr_min',
        ]:
            setattr(self_copy, safe_attr, getattr(cfg, safe_attr))

        return self_copy != cfg

    def bind_io(self):
        r"""
            Add context-specific input/output parameters.
            Has support for re-binding IO, but does _not_ check for shapes, which are assumed to be correct.
            This means we rebind
            - embeddings
            - flags
            - task_modules
            Shapes are hidden sizes for flags/embeddings, and are configured via cfg.
            From this "same cfg" assumption - we will assume that
            `context_project` and `readin` are the same.


            Ideally, we will just bind embedding layers here, but there may be some MLPs.
        """
        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.session, "Session embedding strategy requires session in data"
            if len(self.data_attrs.context.session) == 1:
                logger.warning('Using session embedding strategy with only one session. Expected only if tuning.')
        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.subject, "Subject embedding strategy requires subject in data"
            if len(self.data_attrs.context.subject) == 1:
                logger.warning('Using subject embedding strategy with only one subject. Expected only if tuning.')
        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.array, "Array embedding strategy requires array in data"
            if len(self.data_attrs.context.array) == 1:
                logger.warning('Using array embedding strategy with only one array. Expected only if tuning.')

        # Guardrails (to remove)
        assert len(self.data_attrs.context.task) <= 1, "Only tested for single task"

        # We write the following repetitive logic explicitly to maintain typing
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

        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            self.array_embed = nn.Embedding(len(self.data_attrs.context.array) + 1, self.cfg.array_embed_size, padding_idx=0)
            if self.cfg.array_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.array_embed_size
            elif self.cfg.array_embed_strategy == EmbedStrat.token:
                assert self.cfg.array_embed_size == self.cfg.hidden_size
                self.array_flag = nn.Parameter(torch.zeros(self.cfg.array_embed_size))

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
            assert len(self.data_attrs.context.subject) <= 1, "Only implemented for single subject (likely need padding for mixed batches)"

            for a in self.data_attrs.context.array:
                assert not isinstance(subject_array_registry.query_by_array(a), SortedArrayInfo), "actual mixed readins per session not yet implemented"
            channel_count = sum(
                subject_array_registry.query_by_array(a).get_channel_count() for a in self.data_attrs.context.array
            ) * self.data_attrs.spike_dim

        if project_size is not self.cfg.hidden_size:
            self.context_project = nn.Sequential(
                nn.Linear(project_size, self.cfg.hidden_size),
                nn.ReLU() if self.cfg.activation == 'relu' else nn.GELU(),
            )

        self.readin = nn.Linear(channel_count, self.cfg.hidden_size)

        if self.cfg.task.task == ModelTask.icms_one_step_ahead:
            # TODO add readin for the stim array (similar attr)
            raise NotImplementedError

        self.task_pipelines = nn.ModuleDict({
            k.value: task_modules[k](
                self.backbone.out_size, channel_count, self.cfg, self.data_attrs
            ) for k in [self.cfg.task.task]
        })

    def transfer_io(self, transfer_model: pl.LightningModule):
        r"""
            The logger messages are told from the perspective of a model that is being transferred to (but in practice, this model has been initialized and contains new weights already)
        """
        logger.info("Rebinding IO...")

        transfer_data_attrs: DataAttrs = transfer_model.data_attrs
        transfer_cfg: ModelConfig = transfer_model.cfg
        if self.cfg.task != transfer_cfg.task:
            logger.info(f'Task config updated from {transfer_cfg.task} to {self.cfg.task}')

        def try_transfer(module_name: str):
            if hasattr(self, module_name) and hasattr(transfer_model, module_name):
                if isinstance(getattr(self, module_name), nn.Parameter):
                    getattr(self, module_name).data = getattr(transfer_model, module_name).data
                else:
                    getattr(self, module_name).load_state_dict(getattr(transfer_model, module_name).state_dict())
                logger.info(f'Transferred {module_name} weights.')
            else:
                logger.info(f'New {module_name} weights.')

        if self.data_attrs != transfer_data_attrs:
            def try_transfer_embed(
                embed_name: str, # Used for looking up possibly existing attribute
                new_attrs: List[str],
                old_attrs: List[str] ,
            ) -> nn.Embedding:
                if new_attrs == old_attrs:
                    try_transfer(embed_name)
                if not old_attrs:
                    logger.info(f'New {embed_name} weights.')
                    return
                if not new_attrs:
                    logger.warning(f"No {embed_name} provided in new model despite old model dependency. HIGH CHANCE OF ERROR.")
                    return
                num_reassigned = 0
                embed = getattr(self, embed_name)
                for n_idx, target in enumerate(new_attrs):
                    if target in old_attrs:
                        tar_idx = n_idx + int(embed_name == 'array_embed') # padding offset
                        src_idx = old_attrs.index(target) + int(embed_name == 'array_embed')
                        embed.weight.data[tar_idx] = getattr(transfer_model, embed_name).weight.data[src_idx]
                        num_reassigned += 1
                logger.info(f'Reassigned {num_reassigned} of {len(new_attrs)} {embed_name} weights.')
                if num_reassigned == 0:
                    logger.error(f'No {embed_name} weights reassigned. HIGH CHANCE OF ERROR.')

            try_transfer_embed('session_embed', self.data_attrs.context.session, transfer_data_attrs.context.session)
            try_transfer_embed('subject_embed', self.data_attrs.context.subject, transfer_data_attrs.context.subject)
            try_transfer_embed('array_embed', self.data_attrs.context.array, transfer_data_attrs.context.array)

            try_transfer('session_flag')
            try_transfer('subject_flag')
            try_transfer('array_flag')

        try_transfer('context_project')
        try_transfer('readin')

        for k in self.task_pipelines:
            if k in transfer_model.task_pipelines:
                logger.info(f"Transferred task pipeline {k}.")
                self.task_pipelines[k].load_state_dict(transfer_model.task_pipelines[k].state_dict())

    def freeze_backbone(self):
        logger.info("Freezing backbone.")
        for p in self.backbone.parameters():
            p.requires_grad = False
        # self.backbone.eval() # No, we still want dropout

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
        spikes = torch.as_tensor(
            rearrange(batch[DataKey.spikes], 'b t a c h -> b t a (c h)'),
            dtype=torch.float
        )
        temporal_context = []
        for task in [self.cfg.task.task]:
            temporal_context.extend(self.task_pipelines[task.value].get_temporal_context(batch))
        if self.cfg.task.task == ModelTask.icms_one_step_ahead:
            # Remove final timestep, prepend "initial" quiet recording
            state_in = torch.cat([torch.zeros_like(spikes[:,:1]), spikes[:,:-1]], 1)
        else:
            state_in = spikes
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

        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            import pdb;pdb.set_trace() # TODO vet, also I think the embedding isn't quite right?
            array: torch.Tensor = self.array_embed(batch[MetaKey.array])
            if self.cfg.array_embed_strategy == EmbedStrat.token:
                array = array + self.array_flag
                static_context.append(array)
            elif self.cfg.array_embed_strategy == EmbedStrat.concat:
                array = repeat(array, 'b h -> b t h', t=state_in.shape[1])
                project_context.append(array)

        # TODO support temporal embed + temporal project
        # Do not concat static context - list default is easier to deal with
        # static_context = rearrange(static_context, 't0 b h -> b t0 h') if static_context else None
        if project_context: # someone wanted it
            # B T' H, and we want to merge into B T A H (specifically add T' to each token)
            augmented_tokens, ps = pack([state_in, *project_context], 'b * h')
            augmented_tokens = self.context_project(augmented_tokens)
            state_in = rearrange(augmented_tokens, ps, 'b (t a) h', t=state_in.size(1))
        return state_in, static_context, temporal_context

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # returns backbone features B T A H
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
        ) # B x T x A x H
        if outputs.isnan().any(): # I have no idea why, but something in s61 or s62 throws a nan
            # And strangely, I can't repro by invoking forward again.
            # Leaving for now to see if it happens after padding refactor
            import pdb;pdb.set_trace()
        return outputs

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""
            batch provided contains all configured data_keys and meta_keys
            - The distinction with `forward` is not currently clear, but `_step` is specifically oriented around training.
            Which means it'll fiddle with the payload itself and compute losses

            TODO:
            - Fix: targets are keyed/id-ed per task; there is just a single target variable we're hoping is right
            - ?: Ideally the payloads could be more strongly typed.

            We use modules to control the task-specific readouts, but this isn't multi-task first
            So a shared backbone is assumed. And a single "batch" exists for all paths.
            And moreover, any task-specific _input_ steps (such as masking/shifting) is not well interfaced right now
            (currently overloading `batch` variable, think more clearly either by studying HF repo or considering other use cases)

            Example shapes:
                spikes: B T A C H=1 (C is electrode channel)
                stim: B T C H
                channel_counts: B A (counts per array)
        """
        # TODO figure out how to wrap this IO/ICMS code in a task abstraction
        # ? this logic should maybe be in forward, not separate _step.
        if self.cfg.task.task == ModelTask.icms_one_step_ahead:
            pass
        elif self.cfg.task.task == ModelTask.infill:
            spikes = batch[DataKey.spikes]
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
                DataKey.spikes: spikes,
                'is_masked': is_masked,
                'spike_target': target,
            }
        elif self.cfg.task.task == ModelTask.heldout_decoding:
            pass

        features = self(batch) # B T A H

        # Create outputs for configured task
        batch_out: Dict[str, torch.Tensor] = {}
        for task in [self.cfg.task.task]:
            batch_out.update(self.task_pipelines[task.value](batch, features))
        return batch_out

    @torch.inference_mode()
    def predict(self, batch: Dict[str, torch.Tensor], transform_logrates=True) -> Dict[str, torch.Tensor]:
        r"""
            batch should provide info needed by model. (responsibility of user)
            Output is always batched (for now)
        """
        # there are data keys and meta keys, that might be coming in unbatched
        batch_shapes = {
            DataKey.spikes: '* t a c h',
            DataKey.stim: '* t c h', # TODO review
            MetaKey.session: '*',
            MetaKey.subject: '*',
            MetaKey.array: '* a',
            LENGTH_KEY: '*',
            CHANNEL_KEY: '* a',
        }
        pack_info = {}
        for k in batch:
            batch[k], pack_info[k] = pack([batch[k]], batch_shapes[k])
        features = self(batch)
        batch_out: Dict[str, torch.Tensor] = {}
        for task in [self.cfg.task.task]:
            batch_out.update(self.task_pipelines[task.value](batch, features, compute_metrics=False))
        if transform_logrates:
            if Output.logrates in batch_out:
                batch_out[Output.rates] = self.unpad_and_transform_rates(batch_out[Output.logrates], batch)
            if Output.heldout_logrates in batch_out:
                batch_out[Output.heldout_rates] = self.unpad_and_transform_rates(batch_out[Output.heldout_logrates], batch)
        return batch_out

    def predict_step(
        self, batch, *args, transform_logrates=True, **kwargs
    ):
        return self.predict(batch, transform_logrates=transform_logrates)


    # ==================== Utilities ====================
    def unpad_and_transform_rates(self, logrates: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
            logrates: raw, padded predictions from model, B T A H
        """
        # unpad logrates using LENGTH_KEY and CHANNEL_KEY
        if logrates.shape[2] != 1:
            import pdb;pdb.set_trace()
        assert logrates.shape[2] == 1, "Only single array dim supported for now"
        assert (batch[CHANNEL_KEY] == batch[CHANNEL_KEY].flatten()[0]).all(), "Heterogenuous arrays not supported for now"
        logrates = logrates.unbind()
        if LENGTH_KEY in batch:
            logrates = [l[:b, ...] for l, b in zip(logrates, batch[LENGTH_KEY])]
        if CHANNEL_KEY in batch:
            logrates = [l[:, :, :b[0], ...] for l, b in zip(logrates, batch[CHANNEL_KEY])]
        # try to stack
        if (batch[LENGTH_KEY] == batch[LENGTH_KEY][0]).all():
            logrates = torch.stack(logrates)
        return self.transform_rates(logrates, exp=True, normalize_hz=True).cpu()

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
        def _transform(single: torch.Tensor):
            if exp:
                single = single.exp()
            if normalize_hz:
                single = single / self.data_attrs.bin_size_ms
            return single
        out = logrates
        if isinstance(out, list):
            out = [_transform(o) for o in out]
        else:
            out = _transform(out)
        return out

    # ==================== Optimization ====================
    def common_log(self, metrics, prefix=''):
        self.log(f'{prefix}_loss', metrics['loss'])
        for m in self.cfg.task.metrics:
            self.log(f'{prefix}_{m}', metrics[m])

    def training_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.common_log(metrics, prefix='train')
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.common_log(metrics, prefix='val')
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        metrics = self._step(batch)
        self.common_log(metrics, prefix='test')
        return metrics['loss']

    def configure_optimizers(self):
        scheduler = None
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr_init,
            weight_decay=self.cfg.weight_decay
        )
        if self.cfg.lr_schedule == 'linear_warmup':
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.cfg.lr_ramp_init_factor,
                total_iters=self.cfg.lr_ramp_steps
            )
        elif self.cfg.lr_schedule == 'cosine_warmup':
            scheduler = optim.lr_scheduler.ChainedScheduler([
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.cfg.lr_ramp_init_factor,
                    total_iters=self.cfg.lr_ramp_steps
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.lr_decay_steps,
                    eta_min=self.cfg.lr_min
                ),
            ])
        out = {
            'optimizer': optimizer,
            'monitor': 'val_loss'
        }
        if scheduler is not None:
            out['lr_scheduler'] = scheduler
        return out

# === Model loading ===
# Note - I tried coding this as an override, but PTL `save_hyperparams()` acts up (trying to the save the `self` parameter, apparently) - even when passing explicitly that I just want to save `cfg` and `data_attrs`.
def load_from_checkpoint(checkpoint_path: str, cfg: ModelConfig | None = None, data_attrs: DataAttrs | None = None):
    r"""
        Specifically, model topology is determined by data_attrs.
        data_attrs thus must be saved and loaded with a model to make sense of it.
        However, if we're initializing from another checkpoint, we want to know its data_attrs, but not save it as the new attrs. To avoid doing this while still hooking into PTL `save_hyperparameters()`, we do a manual state_dict transfer of two model instances (one with old and one with new topology.)

        Args:
        - cfg: override, new cfg
        - data_attrs: override, new data_attrs
        cfg level changes are _expected_ to not affect topology,
        BUT TODO e.g. it's unclear if novel weight decay declaration means optimizer is reinitialized?
    """
    old_model = BrainBertInterface.load_from_checkpoint(checkpoint_path)
    if cfg is None and data_attrs is None:
        return old_model
    if cfg is not None:
        if old_model.diff_cfg(cfg):
            raise Exception("Unsupported config diff")
    else:
        cfg = old_model.cfg
    if data_attrs is None:
        data_attrs = old_model.data_attrs
    new_cls = BrainBertInterface(cfg=cfg, data_attrs=data_attrs)
    new_cls.transfer_io(old_model)
    return new_cls
