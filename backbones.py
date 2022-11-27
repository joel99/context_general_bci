from typing import Optional, List
import torch
import torch.nn as nn
import math
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
            ),
            self.cfg.n_layers,
        )
        self.pos_encoder = PositionalEncoding(self.cfg)
        self.dropout_rate = nn.Dropout(self.cfg.dropout)
        # TODO implement concat temporal context via adding a linear layer here;
        # And implement token level etc.

    @property
    def out_size(self):
        return self.cfg.n_state

    def forward(
        self,
        src: torch.Tensor, # T B H
        trial_context: Optional[List[torch.Tensor]] = [], # B x H
        temporal_context: Optional[List[torch.Tensor]] = [], # T x B x H # TODO implement
        padding_mask: Optional[torch.Tensor] = [], # T B
        causal=True
    ) -> torch.Tensor: # T B H
        # testing hypothesis that some src modification is making the nan untraceable?
        src = src + self.pos_encoder(src)
        if causal:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(src.device)
        else:
            src_mask = None
        if len(trial_context) > 0:
            trial_context = torch.cat([c.unsqueeze(0) if c.ndim == 2 else c for c in trial_context], 0)
            src = torch.cat([trial_context, src], 0)
            # allow all to attend to context. Row is attender, col is attended.
            src_mask = torch.cat([
                torch.full((trial_context.shape[0], src_mask.shape[1]), float('-inf'), device=src.device),
                src_mask
            ], 0)
            src_mask = torch.cat([
                torch.full((src_mask.shape[0], trial_context.shape[0]), 0., device=src.device),
                src_mask
            ], 1)
        output = self.encoder(src, src_mask, src_key_padding_mask=padding_mask)
        if len(context) > 0:
            output = output[len(context):]
        output = self.dropout_rate(output)
        return output