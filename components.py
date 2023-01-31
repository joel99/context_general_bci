from typing import Optional, List, Any, Dict, Mapping
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack, repeat, reduce
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

class ReadinCrossAttentionV2(nn.Module):
    r"""
        Motivation: Sorted datasets and DANDI high firing rate is not well integrated. Stitching sort of fixes, but this is expensive (many parameters)
        - We kind of just want to find an alternative to stitching that is not as expensive _per ctx_ because neural data inherently has many ctxs
        - e.g. if 1 session produces ~50 trials, this totally doesn't warrant 16K parameters (e.g. 40 sessions @ 128 channels to 128 hidden size = ~0.6M parameters which is larger than base NDT for ~2K trials),
        - And if 1 session produces 2K trials (as in Monkeys/native), well, we're much better in that case, but this is implausible for human tuning. (still potentially workable with multi-subject transfer but totally empirical)

        Issue with below; projecting each channel to a full value is expensive; using a full key is also unnecessary. (we can have a smaller key to negotiate this initial projection).

        Idea:
        - Still use cross attention with R learned queries ("R" for rank)
        - Use ctx embeddings to update queries.
        - Learn fixed position embeddings of size H (no need to make huge, at most R)
        - Make the value projection just a scalar refactor (for normalization).
        - Max memory consumption will be the B T A C H in Q @ K.
        - In the end we will assemble B T A R, and project into B T A H.
        TODO unclear why we should try this over e.g. just a bottlenecked (compressed) projection; but that's still ~128 x 128 = 16K params per ctx.
        - We can surely learn to identify the "high firing" channels vs the "low firing" channels in more efficient manners.
    """
    def __init__(self, in_count: int, out_count: int, data_attrs: DataAttrs, cfg: ModelConfig):
        raise NotImplementedError

    def forward(self, state_in: torch.Tensor, batch: Dict[str, torch.Tensor], readin=True):
        raise NotImplementedError

class ReadinCrossAttention(nn.Module):
    r"""
        A linear projection (`ReadinMatrix`) is disadvantaged in two ways:
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
            nn.Linear(in_count + cfg.session_embed_size * getattr(cfg, 'session_embed_token_count', 1), cfg.readin_dim),
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
        session_embed = rearrange(
            batch['session'], 'b h -> b 1 1 h' if batch['session'].ndim == 2 else 'b k h -> b 1 1 (k h)'
        )
        return self.readout_project(torch.cat([
        # queries = self.readout_project(torch.cat([
            state_in,
            session_embed.expand(*state_in.size()[:-1], -1),
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
        return pos_embed.transpose(0, 1) if batch_first else pos_embed
        # return rearrange(pos_embed, 't b d -> b t 1 d' if batch_first else 't b d -> t b 1 d')

class SpaceTimeTransformer(nn.Module):
    r"""
        This model transforms temporal sequences of population arrays.
        - There's a spatial component. In early experiments, this was an array dimension.
            - This is still the input shape for now, but we'll likely refactor data to provide tokens.
            - i.e. data stream as <SUBJECT> <ARRAY1> <group 1> <group 2> ... <group N1> <ARRAY2> <group 1> <group 2> ... <group N2> ...
        - We will now refactor into a more generic space dimension.
    """
    def __init__(self, config: TransformerConfig, max_spatial_tokens: int=0):
        super().__init__()
        self.cfg = config
        # self.encoder = torch.jit.script( # In a basic timing test, this didn't appear faster. Built-in transformer likely already very fast.
        enc_layer = nn.TransformerEncoderLayer(
            self.cfg.n_state,
            self.cfg.n_heads,
            dim_feedforward=int(self.cfg.n_state * self.cfg.feedforward_factor),
            dropout=self.cfg.dropout,
            batch_first=True,
            activation=self.cfg.activation,
            norm_first=getattr(self.cfg, 'pre_norm', False),
        )
        if getattr(self.cfg, 'factorized_space_time', False):
            self.space_transformer_encoder = nn.TransformerEncoder(enc_layer, round(self.cfg.n_layers / 2))
            self.time_transformer_encoder = nn.TransformerEncoder(enc_layer, self.cfg.n_layers - round(self.cfg.n_layers / 2))
        else:
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, self.cfg.n_layers)
        self.time_encoder = PositionalEncoding(self.cfg)
        self.dropout_in = nn.Dropout(self.cfg.dropout)
        self.dropout_out = nn.Dropout(self.cfg.dropout)
        # And implement token level etc.
        # if self.cfg.fixup_init:
        #     self.fixup_initialization()
        if self.cfg.transform_space and self.cfg.embed_space:
            n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
            self.space_encoder = nn.Embedding(n_space, self.cfg.n_state)

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
        src: torch.Tensor, # B T A H - embedded already (or possibly B T A G H)
        trial_context: List[torch.Tensor] = [], # T' [B H]
        temporal_context: List[torch.Tensor] = [], # TC' [B T H]
        temporal_padding_mask: Optional[torch.Tensor] = None, # B T
        array_padding_mask: Optional[torch.Tensor] = None, # B A
        causal: bool=True
    ) -> torch.Tensor: # T B H
        r"""
            Each H is a token to be transformed with other T x A tokens.
            Additional T' and T tokens from context are included as well.

            We assume that the provided trial and temporal context is consistently shaped. i.e. any context provided is provided for all samples.
            (So attention masks do not vary across batch)
        """
        # import pdb;pdb.set_trace()

        # === Embeddings ===
        src = self.dropout_in(src)
        # Note that space can essentially use array dim, only logic that needs to account is array_padding_mask
        if self.cfg.transform_space:
            b, t, a, s_a, h = src.size() # s_a for space in array
        else:
            b, t, s, h = src.size() # s for space/array

        # TODO make rotary
        time_embed = rearrange(self.time_encoder(src), 'b t h -> b t 1 h')
        if self.cfg.transform_space:
            src = rearrange(src, 'b t a s_a h -> b t (a s_a) h')
            if self.cfg.embed_space:
                # likely space will soon be given as input or pre-embedded, for now assume range
                space_embed = rearrange(
                    self.space_encoder(torch.arange(src.size(2), device=src.device)
                ), 's h -> 1 1 s h')
                src = src + space_embed
            s = a * s_a
        src = src + time_embed

        # === Masks ===
        def make_src_mask(t: int, s: int, src: torch.Tensor, temporal_context: torch.Tensor | None, trial_context: torch.Tensor | None):
            r"""
                args:
                    temporal_context: b t temp_c h
                    trial_context: b trial_c h

                Produces time major (T*S + TempCtx + TrialCtx, T*S + TempCtx + TrialCtx)
                Use t=1 to produce a space only mask, s=1 to produce a time only mask
            """
            if causal:
                src_mask = nn.Transformer.generate_square_subsequent_mask(t, device=src.device)
                # Add array dimension
                # this tiles such that the flat (t x a) tokens should be t1a1, t1a2, t2a1, t2a2, etc.
                src_mask = rearrange(
                    repeat(src_mask, 't1 t2 -> t1 t2 a1 a2', a1=s, a2=s),
                    't1 t2 a1 a2 -> (t1 a1) (t2 a2)'
                )
            else:
                src_mask = None
            # Update src mask for context. Note that row is attender, col is attended.
            # (For simplicity in construction)
            # Temporal Context is allowed to attend Trial acausally and self causally, but not to src
            # Trial Context is allowed to attend to self acausally, but that's it.
            # Somewhat redundant code structure is to play nice with typing
            if temporal_context is not None: # introduce t * context_num tokens
                if src_mask is None:
                    src_mask = torch.zeros((t * s, t * s), dtype=torch.float, device=src.device) # all attending
                # Since temporal context is expected to be used in a causal cases (ICMS)
                # We provide causal masks; technically there may be a case where spikes should attend all temporal context but can only be achieved indirectly in this setup.
                temporal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=src.device)
                context_num = temporal_context.size(-2)
                temporal_mask = rearrange(
                    repeat(temporal_mask, 't1 t2 -> t1 t2 c1 c2', c1=s + context_num, c2=context_num),
                    't1 t2 c1 c2 -> (t1 c1) (t2 c2)'
                )
                src_mask = F.pad(src_mask, (0, 0, 0, t * context_num), value=float('-inf'))
                src_mask = torch.cat([src_mask, temporal_mask], dim=1)
            if trial_context is not None:
                if src_mask is None:
                    src_mask = torch.zeros((t * s, t * s), dtype=torch.float, device=src.device) # all attending
                src_mask = F.pad(src_mask, (0, 0, 0, trial_context.size(1)), value=float('-inf'))
                src_mask = F.pad(src_mask, (0, trial_context.size(1)), value=0)
            return src_mask

        def make_padding_mask(
            b, t, s, src: torch.Tensor, temporal_context: torch.Tensor | None, array_padding_mask: torch.Tensor | None
        ):
            r"""
                return (b t s) src_key_pad_mask - prevents attending _to_, but not _from_. That should be fine.
                This doesn't include trial context pad (we keep unsquashed so spacetime code can reuse)
            """
            if temporal_padding_mask is not None or array_padding_mask is not None:
                padding_mask = torch.zeros((b, t, s), dtype=torch.bool, device=src.device)

                # Deal with known src padding tokens
                if array_padding_mask is not None:
                    if self.cfg.transform_space:
                        array_padding_mask = repeat(array_padding_mask, 'b a -> b (a s_a)', s_a=s_a)
                    padding_mask |= rearrange(array_padding_mask, 'b a -> b () a')

                # Concat padding mask with temporal context (which augments spatial)
                if temporal_context is not None:
                    padding_mask = F.pad(padding_mask, (0, temporal_context.size(-2)), value=False)
                # Temporal context can be padding, according to temporal padding mask
                if temporal_padding_mask is not None:
                    padding_mask |= rearrange(temporal_padding_mask, 'b t -> b t ()')
                return padding_mask
                # padding_mask = rearrange(padding_mask, 'b t s -> b (t s)')
                # # Trial context is never padded
                # if len(trial_context) > 0:
                #     padding_mask = F.pad(padding_mask, (0, trial_context.size(1)), value=False)
                # return padding_mask
            else:
                return None

        if len(temporal_context) > 0:
            temporal_context = rearrange(temporal_context, 'tc b t h -> b t tc h')
        else:
            temporal_context = None
        if len(trial_context) > 0:
            trial_context = rearrange(trial_context, 'tc b h -> b tc h') # trial context doesn't really belong in either space or time, but can expand along each
        else:
            trial_context = None
        # === Transform ===
        if getattr(self.cfg, 'factorized_space_time', False):
            padding_mask = make_padding_mask(b, t, s, src, temporal_context, array_padding_mask)
            # space first
            space_src = [src]
            if temporal_context is not None:
                space_src.append(temporal_context)
                temporal_context_size = temporal_context.size(-2)
            else:
                temporal_context_size = 0
            if trial_context is not None:
                trial_context_size = trial_context.size(-2)
                space_trial_context = repeat(trial_context, 'b tc h -> b t tc h', t=t)
                space_src.append(space_trial_context)
            else:
                trial_context_size = 0
            space_src, ps = pack(space_src, 'b t * h')
            space_src = rearrange(space_src, 'b t k h -> (b t) k h') # k = s + temp_ctx + trial_ctx

            space_padding_mask = rearrange(padding_mask, 'b t s -> (b t) s')
            space_padding_mask = F.pad(space_padding_mask, (0, trial_context_size), value=False)

            # if there are no context tokens, some spatial sequences may be all padding (e.g. if we pad a trial to a fixed length)
            # in this case, attention will by default produce nans
            # filtering after the fact didn't work (nans still propagated for some unknown reason)
            # so here we filter before hand, which seems to work...
            # note that this empty seq issue only occurs in space; there is always at least one time token to transform
            if not trial_context_size:
                is_empty_seq = torch.all(space_padding_mask, dim=-1)
                # we'll just allow attention in this case, but mask out after the fact to avoid nans
                space_padding_mask[is_empty_seq] = False
            space_src = self.space_transformer_encoder(
                space_src,
                make_src_mask(1, s, src, temporal_context, trial_context),
                src_key_padding_mask=space_padding_mask
            )
            if not trial_context_size:
                space_src = torch.where(rearrange(is_empty_seq, 'bt -> bt 1 1'), 0, space_src)

            space_src, space_trial_context = torch.split(space_src, [s + temporal_context_size, trial_context_size], 1)

            # we can either use this updated context
            # trial context may have updated from spatial transformations (e.g. subject-session interxns)
            # we still want to provide this context to temporal transformer;
            # we can either use this updated context or take trial_context from before
            # we'll prefer to use updated context
            time_src = rearrange(space_src, '(b t) s h -> (b s) t h', b=b)
            if trial_context is not None:
                space_trial_context = reduce(space_trial_context, '(b t) tc h -> b tc h', t=t, reduction='mean')
                time_trial_context = repeat(space_trial_context, 'b tc h -> (b s) tc h', s=s)
                time_src, ps = pack([time_src, time_trial_context], 'b * h')

            time_padding_mask = rearrange(padding_mask, 'b t s -> (b s) t')
            time_padding_mask = F.pad(time_padding_mask, (0, trial_context_size), value=False)

            time_src = self.time_transformer_encoder(
                time_src,
                make_src_mask(t, 1, src, temporal_context, trial_context),
                src_key_padding_mask=time_padding_mask
            )
            if trial_context_size:
                time_src = time_src[:, :-trial_context_size]
            output = rearrange(time_src, '(b s) t h -> b t s h', b=b)
        else:
            contextualized_src = [src]
            if temporal_context is not None:
                contextualized_src.append(temporal_context)
            if trial_context is not None:
                contextualized_src.append(trial_context)
            contextualized_src, ps = pack(contextualized_src, 'b * h') # b [(t a) + (t n) + t'] h

            src_mask = make_src_mask(t, s, src, temporal_context, trial_context)
            # TODO validate - this mask flattening better match the token flattening
            padding_mask = make_padding_mask(b, t, s, src, temporal_context, array_padding_mask)
            padding_mask = rearrange(padding_mask, 'b t s -> b (t s)')
            # Trial context is never padded
            if len(trial_context) > 0:
                padding_mask = F.pad(padding_mask, (0, trial_context.size(-2)), value=False)
            # import pdb;pdb.set_trace()
            # print(t, s, contextualized_src.size(), src_mask.size(), padding_mask.size())
            output = self.transformer_encoder(contextualized_src, src_mask, src_key_padding_mask=padding_mask)
            output = rearrange(output[:, : t * s], 'b (t s) h -> b t s h', t=t, s=s)

        output = self.dropout_out(output)
        if self.cfg.transform_space:
            output = rearrange(output, 'b t (a s_a) h -> b t a s_a h', s_a=s_a)
        return output
