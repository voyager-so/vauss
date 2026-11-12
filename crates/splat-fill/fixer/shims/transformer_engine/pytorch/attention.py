import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _DropExtraState


class DotProductAttention(_DropExtraState, nn.Module):
    def __init__(self, num_attention_heads, kv_channels, num_gqa_groups=None, attention_dropout=0.0,
                 qkv_format="sbhd", attn_mask_type="causal", softmax_scale=None, **kw):
        super().__init__()
        assert attn_mask_type == "no_mask", attn_mask_type
        assert qkv_format in ("bshd", "sbhd"), qkv_format
        self.qkv_format = qkv_format
        self.scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(kv_channels)

    def forward(self, q, k, v, *args, **kw):
        if self.qkv_format == "sbhd":
            q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        b, s = q.shape[0], q.shape[1]
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=self.scale)
        o = o.transpose(1, 2).reshape(b, s, -1)  # [b, s, h*d]
        return o.transpose(0, 1).contiguous() if self.qkv_format == "sbhd" else o


def _rotate_half(x):
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", interleaved=False, fused=False, cu_seqlens=None, **kw):
    """TE semantics: freqs [s, 1, 1, d] are angles; rotate the first d channels, pass the rest through."""
    assert not interleaved and cu_seqlens is None
    seq = t.shape[1] if tensor_format == "bshd" else t.shape[0]
    freqs = freqs[:seq]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [1, s, 1, d]
    rot = freqs.shape[-1]
    tr, tp = t[..., :rot], t[..., rot:]
    cos, sin = torch.cos(freqs).to(t.dtype), torch.sin(freqs).to(t.dtype)
    tr = tr * cos + _rotate_half(tr) * sin
    return torch.cat((tr, tp), dim=-1) if tp.shape[-1] else tr
