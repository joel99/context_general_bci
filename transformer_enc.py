from typing import Optional, List, Any, Dict, Mapping
import copy
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack, repeat, reduce
import logging

from config import TransformerConfig, ModelConfig

# We re-write pytorch transformer due to need for cross attn
# While we're at it, we include the stabilizing norms per https://arxiv.org/pdf/2302.05442.pdf

# TODO we realize during implementation that we can just use pytorch native decoder layer
# We do not bother with norm stabilizer for now

class ContextAugTransformerEncoder(nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, *args, cross_attn_layer=None, **kwargs):
        super().__init__(encoder_layer, num_layers, *args, **kwargs)
        if cross_attn_layer is None:
            self.cross_attn_layers = None
        else:
            self.cross_attn_layer = nn.ModuleList([copy.deepcopy(cross_attn_layer) for _ in range(num_layers)])

    def forward(self, src: Tensor, cross_src: Optional[Tensor]=None, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        # Kill fast path
        output = src
        convert_to_nested = False
        src_key_padding_mask_for_layers = src_key_padding_mask

        for i in range(len(self.layers)):
            # cross first? Why not...
            if self.cross_attn_layers is not None:
                output = self.cross_attn_layers[i](output, cross_src, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output

class ContextCrossAttnLayer(nn.TransformerDecoderLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO implement
    def forward(self, src: Tensor, cross_src: Optional[Tensor]=None, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if cross_src is not None:
            src2 = self.cross_attn(src, cross_src, cross_src, key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        return src