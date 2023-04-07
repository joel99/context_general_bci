r"""
We adapt BrainBertInterface for online decoding specifically.
That is, we remove the flexible interface so we can compile the model as well.
"""

from typing import Tuple, Dict, List, Optional, Any, Mapping
import dataclasses
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack, unpack # baby steps...
from omegaconf import OmegaConf, ListConfig, DictConfig
import logging
from pprint import pformat
from model import unflatten, transfer_cfg, recursive_diff_log, BrainBertInterface

from config import (
    ModelConfig,
    ModelTask,
    Metric,
    Output,
    EmbedStrat,
    DataKey,
    MetaKey,
    Architecture,
)

from data import DataAttrs, LENGTH_KEY, CHANNEL_KEY, COVARIATE_LENGTH_KEY, COVARIATE_CHANNEL_KEY
from subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from components import (
    SpaceTimeTransformer,
    ReadinMatrix,
    ReadinCrossAttention,
    ContextualMLP,
)
from task_io import task_modules, SHUFFLE_KEY, create_temporal_padding_mask, TaskPipeline

logger = logging.getLogger(__name__)

# Assumes FLAT TOKEN SERVING
batch_shapes = {
    DataKey.spikes: '* t token_chan h',
    DataKey.heldout_spikes: '* t c h',
    DataKey.stim: '* t c h', # TODO review
    DataKey.bhvr_vel: '* t h',
    MetaKey.session: '*',
    MetaKey.subject: '*',
    MetaKey.task: '*',
    MetaKey.array: '* a',
    LENGTH_KEY: '*',
    COVARIATE_LENGTH_KEY: '*',
    COVARIATE_CHANNEL_KEY: '*',
    CHANNEL_KEY: '* a', # or '* token'
    DataKey.time: '* t',
    DataKey.position: '* t',
}


class SkinnyBehaviorRegression(TaskPipeline):
    r"""
        Because this is not intended to be a joint task, and backbone is expected to be tuned
        We will not make decoder fancy.
    """

    def __init__(
        self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.decode_cross_attn = True
        self.injector = TemporalTokenInjector(cfg, data_attrs, self.cfg.behavior_target, force_zero_mask=True)
        # TODO reduce into constituent components

        self.time_pad = cfg.transformer.max_trial_length
        self.decoder = SpaceTimeTransformer(
            cfg.transformer,
            max_spatial_tokens=0,
            n_layers=cfg.decoder_layers,
            allow_embed_padding=True,
            context_integration=getattr(cfg, 'decoder_context_integration', 'in_context'),
            embed_space=False
        )
        self.out = nn.Linear(cfg.hidden_size, data_attrs.behavior_dim)
        self.causal = cfg.causal
        self.spacetime = cfg.transform_space
        self.bhvr_lag_bins = round(self.cfg.behavior_lag / data_attrs.bin_size_ms)
        assert self.bhvr_lag_bins >= 0, "behavior lag must be >= 0, code not thought through otherwise"

        if getattr(self.cfg, 'decode_normalizer', ''):
            # See `data_kin_global_stat`
            zscore_path = Path(self.cfg.decode_normalizer)
            assert zscore_path.exists(), f'normalizer path {zscore_path} does not exist'
            self.register_buffer('bhvr_mean', torch.load(zscore_path)['mean'])
            self.register_buffer('bhvr_std', torch.load(zscore_path)['std'])
        else:
            self.bhvr_mean = None
            self.bhvr_std = None

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        # cross attn path
        if self.cfg.decode_strategy == EmbedStrat.token:
            temporal_padding_mask = create_temporal_padding_mask(backbone_features, batch)
            decode_tokens, decode_time, decode_space = self.injector.inject(batch)
            decode_space = torch.zeros_like(decode_space)
            src_time = batch.get(DataKey.time, None)
            if self.causal and self.cfg.behavior_lag_lookahead:
                # allow looking N-bins of neural data into the "future"; we back-shift during the actual decode comparison.
                decode_time = decode_time + self.bhvr_lag_bins
            decoder_input = decode_tokens
            times = decode_time
            positions = decode_space
            other_kwargs = {
                'memory': backbone_features,
                'memory_times': src_time,
                'memory_padding_mask': temporal_padding_mask,
            }
            if temporal_padding_mask is not None:
                temporal_padding_mask = create_temporal_padding_mask(decode_tokens, batch, length_key=COVARIATE_LENGTH_KEY)
            trial_context = []
            for key in ['session', 'subject', 'task']:
                if key in batch and batch[key] is not None:
                    trial_context.append(batch[key].detach()) # Provide context, but hey, let's not make it easier for the decoder to steer the unsupervised-calibrated context
            backbone_features: torch.Tensor = self.decoder(
                decoder_input,
                temporal_padding_mask=temporal_padding_mask,
                trial_context=trial_context,
                times=times,
                positions=positions,
                space_padding_mask=None, # (low pri)
                causal=self.causal,
                **other_kwargs
            )
        bhvr = self.out(backbone_features)
        if self.bhvr_lag_bins:
            bhvr = bhvr[:, :-self.bhvr_lag_bins]
            bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)
        return bhvr



class BrainBertInterfaceDecoder(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(self, cfg: ModelConfig, data_attrs: DataAttrs):
        super().__init__() # store cfg
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.data_attrs = data_attrs
        assert self.data_attrs.max_channel_count % self.cfg.neurons_per_token == 0, "Neurons per token must divide max channel count"
        if self.data_attrs.serve_tokens:
            assert self.cfg.array_embed_strategy == EmbedStrat.none, 'array IDs serving not implemented for spatially tokenized data'
            assert self.cfg.transform_space, 'Transform space must be true if serving (spacetime) tokens'
            assert self.data_attrs.neurons_per_token == self.cfg.neurons_per_token, \
                f"Neurons per token served by data ({self.data_attrs.neurons_per_token}) must match model token size {self.cfg.neurons_per_token}"
        if self.data_attrs.serve_tokens_flat:
            assert self.cfg.transformer.flat_encoder, "Flat encoder must be true if serving flat tokens"
        assert self.cfg.arch == Architecture.ndt, "ndt is all you need"
        if self.cfg.transformer.n_layers == 0: # debug for parity
            self.backbone = nn.Identity()
            self.backbone.out_size = self.cfg.hidden_size
        else:
            self.backbone = SpaceTimeTransformer(
                self.cfg.transformer,
                max_spatial_tokens=data_attrs.max_spatial_tokens,
                debug_override_dropout_out=getattr(cfg.transformer, 'debug_override_dropout_io', False),
                context_integration=getattr(cfg.transformer, 'context_integration', 'in_context'),
                embed_space=cfg.transformer.embed_space,
            )
        self.bind_io()
        self.novel_params: List[str] = [] # for fine-tuning
        num_updates = sum(tp.does_update_root for tp in self.task_pipelines.values())
        assert num_updates <= 1, "Only one task pipeline should update the root"

        if self.cfg.layer_norm_input:
            self.layer_norm_input = nn.LayerNorm(data_attrs.max_channel_count)

        self.token_proc_approx = 0
        self.token_seen_approx = 0
        self.detach_backbone_for_task = False

    def diff_cfg(self, cfg: ModelConfig):
        r"""
            Check if new cfg is different from self.cfg (POV of old model)
        """
        self_copy = self.cfg.copy()
        self_copy = OmegaConf.merge(ModelConfig(), self_copy) # backport novel config
        cfg = OmegaConf.merge(ModelConfig(), cfg)

        # Things that are allowed to change on init (actually most things should be allowed to change, but just register them explicitly here as needed)

        for safe_attr in [
            'decoder_layers', # ! assuming we're freshly initializing, this is kind of not safe
            'decoder_context_integration', # ^
            'dropout',
            'weight_decay',
            'causal',
            'task',
            'lr_init',
            'lr_schedule',
            'lr_ramp_steps',
            'lr_ramp_init_factor',
            'lr_decay_steps',
            'lr_min',
            'accelerate_new_params',
            'tune_decay',
        ]:
            setattr(self_copy, safe_attr, getattr(cfg, safe_attr))
        recursive_diff_log(self_copy, cfg)
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
        if self.cfg.task_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.task, "Task embedding strategy requires task in data"
            if len(self.data_attrs.context.task) == 1:
                logger.warning('Using task embedding strategy with only one task. Expected only if tuning.')

        # We write the following repetitive logic explicitly to maintain typing
        project_size = self.cfg.hidden_size

        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            if self.cfg.session_embed_strategy == EmbedStrat.token and self.cfg.session_embed_token_count > 1:
                self.session_embed = nn.Parameter(torch.randn(len(self.data_attrs.context.session), self.cfg.session_embed_token_count, self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
                self.session_flag = nn.Parameter(torch.randn(self.cfg.session_embed_token_count, self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
            else:
                self.session_embed = nn.Embedding(len(self.data_attrs.context.session), self.cfg.session_embed_size)
                if self.cfg.session_embed_strategy == EmbedStrat.concat:
                    project_size += self.cfg.session_embed_size
                elif self.cfg.session_embed_strategy == EmbedStrat.token:
                    assert self.cfg.session_embed_size == self.cfg.hidden_size
                    if self.cfg.init_flags:
                        self.session_flag = nn.Parameter(torch.randn(self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
                    else:
                        self.session_flag = nn.Parameter(torch.zeros(self.cfg.session_embed_size))

        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            self.subject_embed = nn.Embedding(len(self.data_attrs.context.subject), self.cfg.subject_embed_size)
            if self.cfg.subject_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.subject_embed_size
            elif self.cfg.subject_embed_strategy == EmbedStrat.token:
                assert self.cfg.subject_embed_size == self.cfg.hidden_size
                if self.cfg.init_flags:
                    self.subject_flag = nn.Parameter(torch.randn(self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
                else:
                    self.subject_flag = nn.Parameter(torch.zeros(self.cfg.subject_embed_size))

        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            self.array_embed = nn.Embedding(
                len(self.data_attrs.context.array),
                self.cfg.array_embed_size,
                padding_idx=self.data_attrs.context.array.index('') if '' in self.data_attrs.context.array else None
            )
            self.array_embed.weight.data.fill_(0) # Don't change by default
            if self.cfg.array_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.array_embed_size
            elif self.cfg.array_embed_strategy == EmbedStrat.token:
                assert self.cfg.array_embed_size == self.cfg.hidden_size
                if self.cfg.init_flags:
                    self.array_flag = nn.Parameter(torch.randn(self.data_attrs.max_arrays, self.cfg.array_embed_size) / math.sqrt(self.cfg.array_embed_size))
                else:
                    self.array_flag = nn.Parameter(torch.zeros(self.data_attrs.max_arrays, self.cfg.array_embed_size))

        if self.cfg.task_embed_strategy is not EmbedStrat.none:
            if self.cfg.task_embed_strategy == EmbedStrat.token and getattr(self.cfg, 'task_embed_token_count', 1) > 1:
                self.task_embed = nn.Parameter(torch.randn(len(self.data_attrs.context.task), self.cfg.task_embed_token_count, self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
                self.task_flag = nn.Parameter(torch.randn(self.cfg.task_embed_token_count, self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
            else:
                self.task_embed = nn.Embedding(len(self.data_attrs.context.task), self.cfg.task_embed_size)
                if self.cfg.task_embed_strategy == EmbedStrat.concat:
                    project_size += self.cfg.task_embed_size
                elif self.cfg.task_embed_strategy == EmbedStrat.token:
                    assert self.cfg.task_embed_size == self.cfg.hidden_size
                    if self.cfg.init_flags:
                        self.task_flag = nn.Parameter(torch.randn(self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
                    else:
                        self.task_flag = nn.Parameter(torch.zeros(self.cfg.task_embed_size))

        if project_size is not self.cfg.hidden_size:
            self.context_project = nn.Sequential(
                nn.Linear(project_size, self.cfg.hidden_size),
                nn.ReLU() if self.cfg.activation == 'relu' else nn.GELU(),
            )

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

            # for a in self.data_attrs.context.array:
            #     assert not isinstance(subject_array_registry.query_by_array(a), SortedArrayInfo), "actual mixed readins per session not yet implemented"
            channel_count = sum(
                subject_array_registry.query_by_array(a).get_channel_count() for a in self.data_attrs.context.array
            ) * self.data_attrs.spike_dim

        if self.cfg.transform_space:
            assert self.cfg.spike_embed_style in [EmbedStrat.project, EmbedStrat.token]
            if self.cfg.spike_embed_dim:
                spike_embed_dim = self.cfg.spike_embed_dim
            else:
                assert self.cfg.hidden_size % self.cfg.neurons_per_token == 0, "hidden size must be divisible by neurons per token"
                spike_embed_dim = round(self.cfg.hidden_size / self.cfg.neurons_per_token)
            if self.cfg.spike_embed_style == EmbedStrat.project:
                if getattr(self.cfg, 'debug_project_space', False):
                    self.readin = nn.Linear(channel_count, channel_count)
                else:
                    self.readin = nn.Linear(1, spike_embed_dim)
            elif self.cfg.spike_embed_style == EmbedStrat.token:
                assert self.cfg.max_neuron_count > self.data_attrs.pad_token, "max neuron count must be greater than pad token"
                self.readin = nn.Embedding(self.cfg.max_neuron_count, spike_embed_dim, padding_idx=self.data_attrs.pad_token if self.data_attrs.pad_token else None) # I'm pretty confident we won't see more than 20 spikes in 20ms but we can always bump up
                # Ignore pad token if set to 0 (we're doing 0 pad, not entirely legitimate but may be regularizing)
        else:
            if self.cfg.readin_strategy == EmbedStrat.project or self.cfg.readin_strategy == EmbedStrat.token:
                # Token is the legacy default
                self.readin = nn.Linear(channel_count, self.cfg.hidden_size)
            elif self.cfg.readin_strategy == EmbedStrat.unique_project:
                self.readin = ReadinMatrix(channel_count, self.cfg.
                hidden_size, self.data_attrs, self.cfg)
            elif self.cfg.readin_strategy == EmbedStrat.contextual_mlp:
                self.readin = ContextualMLP(channel_count, self.cfg.hidden_size, self.cfg)
            elif self.cfg.readin_strategy == EmbedStrat.readin_cross_attn:
                self.readin = ReadinCrossAttention(channel_count, self.cfg.hidden_size, self.data_attrs, self.cfg)
        if self.cfg.readout_strategy == EmbedStrat.unique_project:
            self.readout = ReadinMatrix(
                self.cfg.hidden_size,
                self.cfg.readout_dim,
                # self.cfg.readout_dim if getattr(self.cfg, 'readout_dim', 0) else channel_count,
                self.data_attrs,
                self.cfg
            )
            # like PC readout
        elif self.cfg.readout_strategy == EmbedStrat.contextual_mlp:
            self.readout = ContextualMLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg)
            # for simplicity, project out to hidden size - task IO will take care of the other items

        # TODO add readin for the stim array (similar attr)
        # if DataKey.stim in self.data_attrs.<ICMS_ATTR>:
        #   raise NotImplementedError

        def get_target_size(k: ModelTask):
            if k == ModelTask.heldout_decoding:
                # even more hacky - we know only one of these is nonzero at the same time
                return max(
                    self.data_attrs.rtt_heldout_channel_count,
                    self.data_attrs.maze_heldout_channel_count,
                )
            return channel_count
        self.task_pipelines = nn.ModuleDict({
            k.value: task_modules[k](
                self.cfg.hidden_size if task_modules[k].unique_space and self.cfg.readout_strategy is not EmbedStrat.none \
                    else self.backbone.out_size,
                get_target_size(k),
                self.cfg,
                self.data_attrs
            ) for k in self.cfg.task.tasks
        })

    def _wrap_key(self, prefix, key):
        return f'{prefix}.{key}'

    def _wrap_keys(self, prefix, named_params):
        out = []
        for n, p in named_params:
            out.append(self._wrap_key(prefix, n))
        return out

    def transfer_io(self, transfer_model: pl.LightningModule):
        r"""
            The logger messages are told from the perspective of a model that is being transferred to (but in practice, this model has been initialized and contains new weights already)
        """
        logger.info("Rebinding IO...")

        transfer_data_attrs: DataAttrs = transfer_model.data_attrs
        transfer_cfg: ModelConfig = transfer_model.cfg
        if self.cfg.task != transfer_cfg.task:
            logger.info(pformat(f'Task config updating.. (first logged is new config)'))
            recursive_diff_log(self.cfg.task, transfer_cfg.task)
            #  from {transfer_cfg.task} to {self.cfg.task}'))
        def try_transfer(module_name: str):
            if (module := getattr(self, module_name, None)) is not None:
                if (transfer_module := getattr(transfer_model, module_name, None)) is not None:
                    if isinstance(module, nn.Parameter):
                        assert module.data.shape == transfer_module.data.shape
                        # Currently will fail for array flag transfer, no idea what the right policy is right now
                        module.data = transfer_module.data
                    else:
                        if isinstance(module, ReadinMatrix):
                            module.load_state_dict(transfer_module.state_dict(), transfer_data_attrs)
                        else:
                            module.load_state_dict(transfer_module.state_dict())
                    logger.info(f'Transferred {module_name} weights.')
                else:
                    # if isinstance(module, nn.Parameter):
                    #     self.novel_params.append(self._wrap_key(module_name, module_name))
                    # else:
                    #     self.novel_params.extend(self._wrap_keys(module_name, module.named_parameters()))
                    logger.info(f'New {module_name} weights.')
        def try_transfer_embed(
            embed_name: str, # Used for looking up possibly existing attribute
            new_attrs: List[str],
            old_attrs: List[str] ,
        ) -> nn.Embedding:
            if new_attrs == old_attrs:
                try_transfer(embed_name)
                return
            if not hasattr(self, embed_name):
                return
            embed = getattr(self, embed_name)
            if not old_attrs:
                # if isinstance(embed, nn.Parameter):
                #     self.novel_params.append(self._wrap_key(embed_name, embed_name))
                # else:
                #     self.novel_params.extend(self._wrap_keys(embed_name, embed.named_parameters()))
                logger.info(f'New {embed_name} weights.')
                return
            if not new_attrs:
                logger.warning(f"No {embed_name} provided in new model despite old model dependency. HIGH CHANCE OF ERROR.")
                return
            num_reassigned = 0
            def get_param(embed):
                if isinstance(embed, nn.Parameter):
                    return embed
                return getattr(embed, 'weight')
            for n_idx, target in enumerate(new_attrs):
                if target in old_attrs:
                    get_param(embed).data[n_idx] = get_param(getattr(transfer_model, embed_name)).data[old_attrs.index(target)]
                    num_reassigned += 1
            logger.info(f'Reassigned {num_reassigned} of {len(new_attrs)} {embed_name} weights.')
            if num_reassigned == 0:
                logger.warning(f'No {embed_name} weights reassigned. HIGH CHANCE OF ERROR.')
            if num_reassigned < len(new_attrs):
                # There is no non-clunky granular grouping assignment (probably) but we don't need it either
                logger.warning(f'Incomplete {embed_name} weights reassignment, accelerating learning of all.')
                # if isinstance(embed, nn.Parameter):
                #     self.novel_params.append(self._wrap_key(embed_name, embed_name))
                # else:
                #     self.novel_params.extend(self._wrap_keys(embed_name, embed.named_parameters()))
        try_transfer_embed('session_embed', self.data_attrs.context.session, transfer_data_attrs.context.session)
        try_transfer_embed('subject_embed', self.data_attrs.context.subject, transfer_data_attrs.context.subject)
        try_transfer_embed('task_embed', self.data_attrs.context.task, transfer_data_attrs.context.task)
        try_transfer_embed('array_embed', self.data_attrs.context.array, transfer_data_attrs.context.array)

        try_transfer('session_flag')
        try_transfer('subject_flag')
        try_transfer('task_flag')
        try_transfer('array_flag')

        try_transfer('context_project')
        try_transfer('readin')
        try_transfer('readout')

        for k in self.task_pipelines:
            if k in transfer_model.task_pipelines:
                logger.info(f"Transferred task pipeline {k}.")
                self.task_pipelines[k].load_state_dict(transfer_model.task_pipelines[k].state_dict(), strict=False)
            else:
                logger.info(f"New task pipeline {k}.")
                self.novel_params.extend(self._wrap_keys(f'task_pipelines.{k}', self.task_pipelines[k].named_parameters()))

    def forward(
        self,
        spikes: torch.Tensor # T x C x H # ! should be int dtype, double check
    ) -> torch.Tensor: # out is behavior, T x 2
        # do the reshaping yourself
        # ! Assumes this divides evenly
        spikes = spikes.unfold(1, self.cfg.neurons_per_token, self.cfg.neurons_per_token).flatten(-2)
        t, c, h = spikes.size()
        time = repeat(torch.arange(spikes.size(0), device=spikes.device), 't -> (t c)')
        position = repeat(torch.arange(c, device=spikes.device), 'c -> (t c)', t=t)

        # TODO cache session, subject
        session = self.session_embed(torch.zeros(1, dtype=torch.uint8, device=spikes.device)) + self.session_flag
        subject = self.subject_embed(torch.zeros(1, dtype=torch.uint8, device=spikes.device)) + self.subject_flag
        trial_context = [session, subject]

        state_in = self.readin(spikes).flatten(-2).unsqueeze(0)

        features: torch.Tensor = self.backbone(
            state_in,
            trial_context=trial_context,
            temporal_context=[],
            temporal_padding_mask=None, # TODO check this works?
            space_padding_mask=None,
            causal=self.cfg.causal,
            times=time,
            positions=position,
        ) # B x Token x H (flat)

        out = self.task_pipelines[ModelTask.kinematic_decoding.value]
        return outputs

    def _step(self, batch: Dict[str, torch.Tensor], eval_mode=False) -> Dict[str, torch.Tensor]:
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

            Shapes:
                spikes: B T A/S C H=1 (C is electrode channel) (H=1 legacy decision, hypothetically could contain other spike features)
                - if serve_tokens: third dim is space, else it's array
                - if serve tokens flat: Time x A/S is flattened
                stim: B T C H
                channel_counts: B A (counts per array)
        """
        batch_out: Dict[str, torch.Tensor] = {}
        if Output.spikes in self.cfg.task.outputs:
            batch_out[Output.spikes] = batch[DataKey.spikes][..., 0]
        for task in self.cfg.task.tasks:
            self.task_pipelines[task.value].update_batch(batch, eval_mode=eval_mode)
        features = self(batch) # B T S H
        if self.cfg.log_backbone_norm:
            # expected to track sqrt N. If it's not, then we're not normalizing properly
            self.log('backbone_norm', torch.linalg.vector_norm(
                features.flatten(0, -2), dim=-1
            ).mean(), on_epoch=True, batch_size=features.size(0))

        if not self.cfg.transform_space:
            # no unique strategies will be tried for spatial transformer (its whole point is ctx-robustness)
            if self.cfg.readout_strategy == EmbedStrat.mirror_project:
                unique_space_features = self.readin(features, batch, readin=False)
            elif self.cfg.readout_strategy in [EmbedStrat.unique_project, EmbedStrat.contextual_mlp]:
                unique_space_features = self.readout(features, batch)

        # Create outputs for configured task
        running_loss = 0
        task_order = self.cfg.task.tasks
        if self.cfg.task.kl_lambda > 0 and ModelTask.kinematic_decoding in self.cfg.task.tasks:
            task_order = [ModelTask.kinematic_decoding]
            for t in self.cfg.task.tasks:
                if t != ModelTask.kinematic_decoding:
                    task_order.append(t)
        if getattr(self.cfg.task, 'decode_use_shuffle_backbone', False):
            task_order = [ModelTask.shuffle_infill]
            for t in self.cfg.task.tasks:
                if t != ModelTask.shuffle_infill:
                    task_order.append(t)
        for i, task in enumerate(task_order):
            pipeline_features = unique_space_features if self.task_pipelines[task.value].unique_space and self.cfg.readout_strategy is not EmbedStrat.none else features
            if 'infill' not in task.value and self.detach_backbone_for_task:
                pipeline_features = pipeline_features.detach()
            update = self.task_pipelines[task.value](
                batch,
                pipeline_features,
                eval_mode=eval_mode
            )
            for k in update:
                if 'update' in str(k):
                    if k == 'update_features':
                        features = update[k]
                    batch[k] = update[k]
                else:
                    batch_out[k] = update[k]
            if 'loss' in update and self.cfg.task.task_weights[i] > 0:
                batch_out[f'{task.value}_loss'] = update['loss']
                running_loss = running_loss + self.cfg.task.task_weights[i] * update['loss']
        batch_out['loss'] = running_loss
        return batch_out

    @torch.inference_mode()
    def predict(
        self, batch: Dict[str, torch.Tensor], transform_logrates=True, mask=False,
        eval_mode=True,
        # eval_mode=False,
    ) -> Dict[str, torch.Tensor]:
        r"""
            Severely stripped for online decoding
        """
        for k in batch:
            batch[k], _ = pack([batch[k]], batch_shapes[k])
        batch_out: Dict[str, torch.Tensor] = {}
        features = self(batch)
        for task in self.cfg.task.tasks:
            update = self.task_pipelines[task.value](
                batch,
                features,
                compute_metrics=False,
                eval_mode=eval_mode
            )
            batch_out[k] = update[k]
        return batch_out

    def predict_step(
        self, batch, *args, transform_logrates=True, mask=True, **kwargs
        # self, batch, *args, transform_logrates=True, mask=False, **kwargs
    ):
        return self.predict(batch, transform_logrates=transform_logrates, mask=mask)


def transfer_model(old_model: BrainBertInterface, new_cfg: ModelConfig, new_data_attrs: DataAttrs):
    r"""
        Transfer model to new cfg and data_attrs.
        Intended to be for inference
    """
    if new_cfg is None and new_data_attrs is None:
        return old_model
    if new_cfg is not None:
        transfer_cfg(src_cfg=old_model.cfg, target_cfg=new_cfg)
        if old_model.diff_cfg(new_cfg):
            raise Exception("Unsupported config diff")
    else:
        new_cfg = old_model.cfg
    if new_data_attrs is None:
        new_data_attrs = old_model.data_attrs
    new_cls = BrainBertInterfaceDecoder(cfg=new_cfg, data_attrs=new_data_attrs)
    new_cls.backbone.load_state_dict(old_model.backbone.state_dict())
    new_cls.transfer_io(old_model)
    return new_cls
