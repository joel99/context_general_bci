from typing import Tuple, Dict, List, Optional, Any
import dataclasses
import numpy as np
import torch
from torch import nn, optim
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack # baby steps...

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
        self.data_attrs = None
        self.bind_io(data_attrs)

    def diff_cfg(self, cfg: ModelConfig):
        r"""
            Check if cfg is different from current cfg (used when loading)
        """
        self_copy = self.cfg.copy()
        ref_copy = cfg.copy()
        self_copy.task = TaskConfig()
        ref_copy.task = TaskConfig()
        return self_copy != ref_copy

    def bind_io(self, data_attrs: DataAttrs, task_cfg: Optional[TaskConfig] = None):
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
            assert data_attrs.context.session, "Session embedding strategy requires session in data"
            if len(data_attrs.context.session) == 1:
                logger.warn('Using session embedding strategy with only one session. Expected only if tuning.')
        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            assert data_attrs.context.subject, "Subject embedding strategy requires subject in data"
            if len(data_attrs.context.subject) == 1:
                logger.warn('Using subject embedding strategy with only one subject. Expected only if tuning.')
        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            assert data_attrs.context.array, "Array embedding strategy requires array in data"
            if len(data_attrs.context.array) == 1:
                logger.warn('Using array embedding strategy with only one array. Expected only if tuning.')
        old_attrs = None
        if self.data_attrs is not None: # IO already exists
            if self.data_attrs == data_attrs:
                logger.info('Identical IO detected, skipping IO rebind')
            else:
                r"""
                    The other modules are: flags, embedding modules, readin, context_project, task_pipelines.
                    TODO address in turn
                """
                logger.info("Rebinding IO...")
                old_attrs = self.data_attrs

        self.data_attrs = data_attrs

        # Guardrails (to remove)
        assert len(self.data_attrs.context.task) <= 1, "Only tested for single task"
        assert self.cfg.array_embed_strategy == EmbedStrat.none, "Array embed strategy not yet implemented"

        def init_and_maybe_transfer_embed(
            new_attrs: List[str],
            embed_size: int,
            embed_name: str, # Used for looking up possibly existing attribute
            old_attrs: List[str] | None = None,
        ) -> nn.Embedding:
            if new_attrs == old_attrs:
                logger.info(f'Identical {embed_name} detected, skipping rebind.')
                assert getattr(self, embed_name) is not None

            embed = nn.Embedding(len(new_attrs) + int(embed_name == 'array_embed'), embed_size)
            # # +1 is for padding (i.e. self.array_embed[-1] = padding)

            if not hasattr(self, embed_name):
                logger.info(f'Initializing {embed_name} from scratch.')
                return embed

            num_reassigned = 0
            for n_idx, target in enumerate(new_attrs):
                if target in old_attrs:
                    embed.weight.data[n_idx] = getattr(self, embed_name).weight.data[old_attrs.index(target)]
                    num_reassigned += 1
            logger.info(f'Reassigned {num_reassigned} of {len(new_attrs)} {embed_name} weights.')
            if num_reassigned == 0:
                logger.error(f'No {embed_name} weights reassigned. HIGH CHANCE OF ERROR.')
            if embed_name == 'array_embed':
                embed.weight.data[-1] = getattr(self, embed_name).weight.data[-1] # padding
            return embed

        def init_or_transfer_flag(flag_size, flag_name):
            return getattr(self, flag_name, nn.Parameter(torch.zeros(flag_size)))

        # We write the following repetitive logic explicitly to maintain typing
        project_size = self.cfg.hidden_size

        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            self.session_embed = init_and_maybe_transfer_embed(
                self.data_attrs.context.session,
                self.cfg.session_embed_size,
                'session_embed',
                old_attrs.context.session if old_attrs else None,
            )
            if self.cfg.session_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.session_embed_size
            elif self.cfg.session_embed_strategy == EmbedStrat.token:
                assert self.cfg.session_embed_size == self.cfg.hidden_size
                self.session_flag = init_or_transfer_flag(self.cfg.session_embed_size, 'session_flag')

        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            self.subject_embed = init_and_maybe_transfer_embed(
                self.data_attrs.context.subject,
                self.cfg.subject_embed_size,
                'subject_embed',
                old_attrs.context.subject if old_attrs else None,
            )
            if self.cfg.subject_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.subject_embed_size
            elif self.cfg.subject_embed_strategy == EmbedStrat.token:
                assert self.cfg.subject_embed_size == self.cfg.hidden_size
                self.subject_flag = init_or_transfer_flag(self.cfg.subject_embed_size, 'subject_flag')

        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            self.array_embed = init_and_maybe_transfer_embed(
                self.data_attrs.context.array,
                self.cfg.array_embed_size,
                'array_embed',
                old_attrs.context.array if old_attrs else None,
            )
            if self.cfg.array_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.subject_embed_size
            elif self.cfg.array_embed_strategy == EmbedStrat.token:
                assert self.cfg.array_embed_size == self.cfg.hidden_size
                self.array_flag = init_or_transfer_flag(self.cfg.array_embed_size, 'array_flag')

        if project_size is not self.cfg.hidden_size:
            self.context_project = nn.Sequential(
                nn.Linear(project_size, self.cfg.hidden_size),
                nn.ReLU()
            )
        else:
            self.context_project = None

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

        if task_cfg is not None and self.cfg.task != task_cfg:
            logger.info(f'Updating task config from {self.cfg.task} to {task_cfg}')
            self.cfg.task = task_cfg

        if self.cfg.task.task == ModelTask.icms_one_step_ahead:
            # TODO add readin for the stim array (similar attr)
            raise NotImplementedError

        task_pipelines = nn.ModuleDict({
            k.value: task_modules[k](
                self.backbone.out_size, channel_count, self.cfg
            ) for k in [self.cfg.task.task]
        })
        if hasattr(self, 'task_pipelines'):
            for k in task_pipelines:
                if k in self.task_pipelines:
                    task_pipelines[k].load_state_dict(self.task_pipelines[k].state_dict())
        self.task_pipelines = task_pipelines


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

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # returns backbone features B T A H
        import pdb;pdb.set_trace()
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
        target = None
        # TODO figure out how to wrap this IO/ICMS code in a task abstraction
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
                DataKey.spikes: torch.as_tensor(spikes, dtype=torch.float),
                'is_masked': is_masked,
                'spike_target': target,
            }


        features = self(batch) # B T A H

        # Create outputs for configured task
        batch_out: Dict[str, torch.Tensor] = {}
        for task in [self.cfg.task.task]:
            batch_out.update(self.task_pipelines[task.value](batch, features))
        return batch_out

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        return self(x)

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

    # ==================== Optimization ====================
    def predict_step(
        self, batch
    ):
        return self.predict(batch)

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
        return {
            'optimizer': optim.AdamW(
                self.parameters(),
                lr=self.cfg.lr_init,
                weight_decay=self.cfg.weight_decay
            ),
            'monitor': 'val_loss'
        }