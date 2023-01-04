from typing import Optional, List, Any, Dict, Mapping
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack, repeat
import logging

from config import TransformerConfig, ModelConfig
from data import DataAttrs, MetaKey

logger = logging.getLogger(__name__)

class ReadinMatrix(nn.Module):
    r"""
        Linear projection to transform input population to canonical (probably PC-related) input.
        Optional rank bottleneck (`readin_compress`)
    """
    def __init__(self, in_count: int, out_count: int, data_attrs: DataAttrs, cfg: ModelConfig):
        super().__init__()
        self.contexts = data_attrs.context.session # ! currently assuming no session overlap
        self.compress = cfg.readin_compress
        self.unique_readin = nn.Parameter(
            init.kaiming_uniform_(
                torch.empty(len(self.contexts), in_count, cfg.readin_dim if self.compress else out_count),
                a=math.sqrt(5)
            )
        )
        if self.compress:
            self.project = nn.Parameter(
                init.kaiming_uniform_(
                    torch.empty(cfg.readin_dim, out_count),
                )
            )


    def forward(self, state_in: torch.Tensor, batch: Dict[str, torch.Tensor], readin=True):
        r"""
            state_in: B T A H
        """
        # use session (in lieu of context) to index readin parameter (b x in x out)
        readin_matrix = torch.index_select(self.unique_readin, 0, batch[MetaKey.session])
        if readin:
            state_in = torch.einsum('btai,bih->btah', state_in, readin_matrix)
            if self.compress:
                state_in = torch.matmul(state_in, self.project)
        else: # readout
            readin_matrix = rearrange(readin_matrix, 'b i h -> b h i')
            if self.compress:
                state_in = torch.matmul(state_in, self.project.T) # b t a h x h readin
            state_in = torch.einsum('btah,bhi->btai', state_in, readin_matrix)
        return state_in

    def load_state_dict(self, transfer_state_dict: Mapping[str, Any], transfer_attrs: DataAttrs):
        state_dict = {}
        if self.compress:
            state_dict['project'] = transfer_state_dict['project']
        num_reassigned = 0
        current_state = self.state_dict()['unique_readin']
        for n_idx, target in enumerate(self.contexts):
            if target in transfer_attrs.context.session:
                s_idx = transfer_attrs.context.session.index(target)
                current_state[n_idx] = transfer_state_dict[f'unique_readin'][s_idx]
                num_reassigned += 1
        logger.info(f'Loaded {num_reassigned} of {len(self.contexts)} readin matrices.')
        state_dict['unique_readin'] = current_state # unnecessary?
        return super().load_state_dict(state_dict)

class ReadinCrossAttention(nn.Module):
    r"""
        A linear projection is disadvantaged in two ways:
        - couples value with relevance
        - requires a separate projection (~Channel x Hidden) for each context
            - extra parameters may be statistically inefficient (though computational footprint is negligible)

        We thus apply a cross-attention readin strategy, which outsources context-specific parameters to the context embeddings.
        Hopefully they have enough capacity. Also, this module has high memory costs (several GB) due to hidden-vector per channel in value computation
        # ! Actually, wayy too much memory. We actually go over 14G just scaling on Indy.
        # Also, initial testing shows little promise in small-scale Maze nlb.

        - individual channels get learned position embeddings
        - position embedding + context specific embedding (e.g. session embed) concat and project to
            - key
            - value
            - TODO add more context embeds (subject, array)
            - TODO decouple these context embeds with global step context embed
        - global query vectors of size H
            - TODO develop task-queries

        Readout strategies:
        - take backbone, pos embed and context embed, project to H
        - TODO should readout include task embed? (maybe as a global pre-transform, to be symmetric)
    """
    def __init__(self, in_count: int, out_count: int, data_attrs: DataAttrs, cfg: ModelConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(cfg.readin_dim))
        self.channel_position_embeds = nn.Parameter(init.kaiming_uniform_(torch.empty(in_count, cfg.readin_dim), a=math.sqrt(5)))
        self.scale = math.sqrt(cfg.readin_dim)
        self.key_project = nn.Sequential(
            nn.Linear(cfg.readin_dim + cfg.session_embed_size, cfg.readin_dim),
            nn.GELU(),
            nn.Linear(cfg.readin_dim, cfg.readin_dim)
        )
        self.value_project = nn.Sequential(
            nn.Linear(1 + cfg.readin_dim + cfg.session_embed_size, cfg.readin_dim),
            nn.GELU(),
            nn.Linear(cfg.readin_dim, out_count)
        )

    def forward(self, state_in: torch.Tensor, session: torch.Tensor, subject: torch.Tensor, array: torch.Tensor):
        r"""
            state_in: B T A C
            session: B x H
            subject: B x H
            array: B x A x H
        """
        b, t, a, c = state_in.size()
        h = session.size(-1)
        r = self.channel_position_embeds.size(-1)
        keys = self.key_project(torch.cat([
            rearrange(self.channel_position_embeds, 'c r -> 1 c r').expand(b, c, r), # add batch dim
            rearrange(session, 'b h -> b 1 h').expand(b, c, h) # add input-channel dim
        ], dim=-1)) # b c r
        values = self.value_project(torch.cat([
            rearrange(self.channel_position_embeds, 'c r -> 1 1 1 c r').expand(b, t, a, c, r), # add B T A
            rearrange(session, 'b h -> b 1 1 1 h').expand(b, t, a, c, h), # add T A C dim
            state_in.unsqueeze(-1),
        ], dim=-1)) # b t a c h

        # perform cross attention
        scores = torch.einsum('bcr, r->bc', keys, self.query) / self.scale
        normalized_scores = F.softmax(scores, dim=-1)

        state_in = torch.einsum('bc, btach -> btah', normalized_scores, values) # b q c x b t a c h
        return state_in

class ContextualMLP(nn.Module):
    def __init__(self, in_count: int, out_count: int, cfg: ModelConfig):
        super().__init__()
        # self.channel_position_embeds = nn.Parameter(init.kaiming_uniform_(torch.empty(out_count, cfg.readin_dim), a=math.sqrt(5)))
        self.readout_project = nn.Sequential(
            nn.Linear(in_count + cfg.session_embed_size, cfg.readin_dim),
            nn.GELU(),
            nn.Linear(cfg.readin_dim, out_count)
            # nn.Linear(cfg.readin_dim, cfg.readin_dim)
        )

    def forward(self, state_in: torch.Tensor, batch: Dict[str, torch.Tensor]):
        r"""
            state_in: B T A H
            session: B x H
            subject: B x H
            array: B x A x H

            out: B T A H (or C)
        """

        return self.readout_project(torch.cat([
        # queries = self.readout_project(torch.cat([
            state_in,
            rearrange(batch['session'], 'b h -> b 1 1 h').expand(*state_in.size()[:-1], -1),
        ], -1))
        # To show up in a given index, a query must have a high score against that index embed
        # return torch.einsum('btah, ch -> btac', queries, self.channel_position_embeds)



class PositionalEncoding(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        position = torch.arange(0, cfg.max_trial_length, dtype=torch.float).unsqueeze(1)
        self.learnable = cfg.learnable_position
        # if self.learnable:
        #     self.register_buffer('pe', position.long())
        #     self.pos_embedding = nn.Embedding(cfg.max_trial_length, cfg.n_state)
        # else:
        pe = torch.zeros(cfg.max_trial_length, cfg.n_state)
        div_term = torch.exp(torch.arange(0, cfg.n_state, 2).float() * (-math.log(10000.0) / cfg.n_state))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) # t x 1 x d
        self.register_buffer('pe', pe)
        if self.learnable:
            self.pe = nn.Parameter(self.pe)

    def forward(self, x: torch.Tensor, batch_first=True):
        # if self.learnable:
        #     pos_embed = self.pos_embedding(self.pe) # t 1 d
        # else:
        #     pos_embed = self.pe[:x.size(1 if batch_first else 0), :]
        pos_embed = self.pe[:x.size(1 if batch_first else 0), :]
        return rearrange(pos_embed, 't b d -> b t 1 d' if batch_first else 't b d -> t b 1 d')

class TemporalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.cfg = config
        # self.encoder = torch.jit.script( # In a basic timing test, this didn't appear faster. Built-in transformer likely already very fast.
        self.encoder = (
                nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    self.cfg.n_state,
                    self.cfg.n_heads,
                    dim_feedforward=int(self.cfg.n_state * self.cfg.feedforward_factor),
                    dropout=self.cfg.dropout,
                    batch_first=False, # we use this to stick to pytorch defaults. Who knows if it's more efficient internally? But closer to docs.
                    activation=self.cfg.activation,
                    norm_first=getattr(self.cfg, 'pre_norm', False),
                ),
                self.cfg.n_layers,
            )
        )
        self.pos_encoder = PositionalEncoding(self.cfg)
        self.dropout_in = nn.Dropout(self.cfg.dropout)
        self.dropout_out = nn.Dropout(self.cfg.dropout)
        # And implement token level etc.
        if getattr(self.cfg, 'fixup_init', False):
        # if self.cfg.fixup_init:
            self.fixup_initialization()

    def fixup_initialization(self):
        r"""
        http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        temp_state_dic = {}
        en_layers = self.cfg.n_layers

        for l in self.encoder.layers:
            for name, param in l.named_parameters():
                if name in ["linear1.weight",
                            "linear2.weight",
                            "self_attn.out_proj.weight",
                            ]:
                    temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param
                elif name in ["self_attn.v_proj.weight",]:
                    temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * (param * (2**0.5))
            l.load_state_dict(temp_state_dic, strict=False)

    @property
    def out_size(self):
        return self.cfg.n_state

    def forward(
        self,
        src: torch.Tensor, # B T A H
        trial_context: List[torch.Tensor] = [], # T' [B H]
        temporal_context: List[torch.Tensor] = [], # TC' [B T H]
        temporal_padding_mask: Optional[torch.Tensor] = None, # B T
        array_padding_mask: Optional[torch.Tensor] = None, # B A
        causal: bool=True
    ) -> torch.Tensor: # T B H
        # testing hypothesis that some src modification is making the nan untraceable?
        r"""
            Each H is a token to be transformed with other T x A tokens.
            Additional T' and T tokens from context are included as well.

            We assume that the provided trial and temporal context is consistently shaped. i.e. any context provided is provided for all samples.
            (So attention masks do not vary across batch)
        """
        # import pdb;pdb.set_trace() # ! TODO need to test mixed multi-array settings
        b, t, a, h = src.size()
        src = self.dropout_in(src)
        src = src + self.pos_encoder(src) # TODO make relative

        contextualized_src, ps = pack([
            src,
            *temporal_context,
            *trial_context
        ], 'b * h') # b [(t a) + (t n) + t'] h

        contextualized_src = rearrange(contextualized_src, 'b x h -> x b h')

        # src mask
        if causal:
            # import pdb;pdb.set_trace() # untested codepath
            src_mask = nn.Transformer.generate_square_subsequent_mask(t, device=src.device)
            # Add array dimension
            src_mask = rearrange(
                repeat(src_mask, 't1 t2 -> t1 t2 a1 a2', a1=a, a2=a),
                't1 t2 a1 a2 -> a1 t1 a2 t2'
            )
        else:
            src_mask = None

        # Update src mask for context. Note that row is attender, col is attended.
        # (For simplicity in construction)
        # Temporal Context is allowed to attend Trial acausally and self causally, but not to src
        # Trial Context is allowed to attend to self acausally, but that's it.
        # Somewhat redundant code structure is to play nice with typing
        if len(temporal_context) > 0:
            if src_mask is None:
                src_mask = torch.zeros((t * a, t * a), dtype=torch.float, device=src.device) # all attending
            # Since temporal context is expected to be used in a causal cases (ICMS)
            # We provide causal masks; technically there may be a case where spikes should attend all temporal context but can only be achieved indirectly in this setup.
            temporal_context: torch.Tensor = rearrange(temporal_context, 'c b t h -> b t c h')
            temporal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=src.device)
            context_num = temporal_context.size(2)
            temporal_mask = rearrange(
                repeat(temporal_mask, 't1 t2 -> t1 t2 c1 c2', c1=context_num+a, c2=context_num),
                't1 t2 c1 c2 -> c1 t1 c2 t2'
            )
            src_mask = F.pad(src_mask, (0, 0, 0, t * context_num), value=float('-inf'))
            # src_mask = torch.cat([
            #     src_mask,
            #     torch.full((t * context_num, t * a), float('-inf'), dtype=torch.float, device=src.device),
            # ], dim=0)
            src_mask = torch.cat([
                src_mask,
                temporal_mask,
            ], dim=1)
        if len(trial_context) > 0:
            if src_mask is None:
                src_mask = torch.zeros((t * a, t * a), dtype=torch.float, device=src.device) # all attending
            trial_context: torch.Tensor = rearrange(trial_context, 't0 b h -> b t0 h')
            src_mask = F.pad(src_mask, (0, 0, 0, trial_context.size(1)), value=float('-inf'))
            src_mask = F.pad(src_mask, (0, trial_context.size(1)), value=0)

        # TODO validate - this mask flattening better match the token flattening

        # padding mask
        if temporal_padding_mask is not None or array_padding_mask is not None:
            padding_mask = torch.zeros((b, t, a), dtype=torch.bool, device=src.device)

            # Update padding mask for context
            # Temporal context can be padded, according to temporal padding mask
            if len(temporal_context) > 0:
                padding_mask = F.pad(padding_mask, (0, context_num), value=False)

            if temporal_padding_mask is not None:
                padding_mask |= rearrange(temporal_padding_mask, 'b t -> b t ()')

            if array_padding_mask is not None:
                padding_mask |= rearrange(array_padding_mask, 'b a -> b () a')

            padding_mask = rearrange(padding_mask, 'b t a -> b (t a)')

            # Trial context is never padded
            if len(trial_context) > 0:
                padding_mask = F.pad(padding_mask, (0, trial_context.size(1)), value=False)
        else:
            padding_mask = None
        output = self.encoder(contextualized_src, src_mask, src_key_padding_mask=padding_mask)
        output = rearrange(output[: t * a], '(t a) b h -> b t a h', t=t, a=a)
        output = self.dropout_out(output)
        return output
