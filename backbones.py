from typing import Optional, List
import torch
import torch.nn as nn
import math
from einops import rearrange, pack, unpack, repeat

from config import TransformerConfig

class PositionalEncoding(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=cfg.dropout)
        position = torch.arange(0, cfg.max_trial_length, dtype=torch.float).unsqueeze(1)
        self.learnable = cfg.learnable_position
        if self.learnable:
            self.register_buffer('pe', position.long())
            self.pos_embedding = nn.Embedding(cfg.max_trial_length, cfg.n_state) # So maybe it's here...?
        else:
            pe = torch.zeros(cfg.max_trial_length, cfg.n_state)
            div_term = torch.exp(torch.arange(0, cfg.n_state, 2).float() * (-math.log(10000.0) / cfg.n_state))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1) # t x 1 x d
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        if self.learnable:
            x = x + self.pos_embedding(self.pe) # t x 1 x d
        else:
            x = x + self.pe[:x.size(0), :] # t x 1 x d, # t x b x d
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.cfg = config
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.cfg.n_state,
                self.cfg.n_heads,
                dim_feedforward=self.cfg.n_state * self.cfg.feedforward_factor,
                dropout=self.cfg.dropout,
                batch_first=False, # we use this to stick to pytorch defaults. Who knows if it's more efficient internally? But closer to docs.
            ),
            self.cfg.n_layers,
        )
        self.pos_encoder = PositionalEncoding(self.cfg)
        self.dropout_rate = nn.Dropout(self.cfg.dropout)
        # And implement token level etc.

    @property
    def out_size(self):
        return self.cfg.n_state

    def forward(
        self,
        src: torch.Tensor, # B T A H
        trial_context: List[torch.Tensor] = [], # B T' H
        temporal_context: List[torch.Tensor] = [], # B T H # TODO implement
        temporal_padding_mask: Optional[torch.Tensor] = None, # B T
        array_padding_mask: Optional[torch.Tensor] = None, # B A
        causal=True
    ) -> torch.Tensor: # T B H
        # testing hypothesis that some src modification is making the nan untraceable?
        r"""
            Each H is a token to be transformed with other T x A tokens.
            Additional T' and T tokens from context are included as well.

            We assume that the provided trial and temporal context is consistently shaped. i.e. any context provided is provided for all samples.
            (So attention masks do not vary across batch)
        """
        b, t, a, h = src.size()
        src = src + self.pos_encoder(src) # TODO make relative
        contextualized_src, ps = pack([
            src,
            *temporal_context,
            *trial_context
        ], 'b * h') # b [(t a) + (t n) + t'] h

        contextualized_src = rearrange(contextualized_src, 'b x h -> x b h')

        # src mask
        if causal:
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
                src_mask = torch.zeros_like((t * a, (t * a)), dtype=torch.float, device=src.device) # all attending
            # Since temporal context is expected to be used in a causal cases (ICMS)
            # We provide causal masks; technically there may be a case where spikes should attend all temporal context but can only be achieved indirectly in this setup.
            temporal_context: torch.Tensor = rearrange(temporal_context, 'c b t h -> b t c h')
            temporal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=src.device)
            context_num = temporal_context.size(2)
            temporal_mask = rearrange(
                repeat(temporal_mask, 't1 t2 -> t1 t2 c1 c2', c1=context_num+a, c2=context_num),
                't1 t2 c1 c2 -> c1 t1 c2 t2'
            )
            src_mask = torch.cat([
                src_mask,
                torch.full((t * context_num, t * a), float('-inf'), dtype=torch.float, device=src.device),
            ], dim=0)
            src_mask = torch.cat([
                src_mask,
                temporal_mask,
            ], dim=1)
        if len(trial_context) > 0:
            if src_mask is None:
                src_mask = torch.zeros_like((t * a, (t * a)), dtype=torch.float) # all attending
            trial_context: torch.Tensor = rearrange(trial_context, 't0 b h -> b t0 h')
            src_mask = torch.cat([
                src_mask,
                torch.full((trial_context.size(1), src_mask.size(1)), float('-inf'), device=src_mask.device)
            ], dim=0)
            src_mask = torch.cat([
                src_mask,
                torch.full((trial_context.size(1), src_mask.size(0)), 0, device=src_mask.device)
            ], dim=1)

        # TODO validate - this mask flattening better match the token flattening


        # padding mask
        if temporal_padding_mask is not None or array_padding_mask is not None:
            padding_mask = torch.zeros((b, t, a), dtype=torch.bool, device=src.device)

            # Update padding mask for context
            # Temporal context can be padded, according to temporal padding mask
            if len(temporal_context) > 0:
                padding_mask = torch.cat([
                    padding_mask,
                    torch.zeros((b, t, context_num), dtype=torch.bool, device=src.device)
                ], dim=2)

            if temporal_padding_mask is not None:
                padding_mask |= temporal_padding_mask # B T

            if array_padding_mask is not None:
                padding_mask |= rearrange(array_padding_mask, 'b a -> b () a')

            padding_mask = rearrange(padding_mask, 'b t a -> b (t a)')

            # Trial context is never padded
            if len(trial_context) > 0:
                padding_mask = torch.cat([
                    padding_mask,
                    torch.zeros((b, trial_context.size(1)), dtype=torch.bool, device=padding_mask.device)
                ])
        else:
            padding_mask = None

        output = self.encoder(contextualized_src, src_mask, src_key_padding_mask=padding_mask)
        output = rearrange(output[: t * a], '(t a) b h -> b t a h', t=t, a=a)
        output = self.dropout_rate(output)
        return output