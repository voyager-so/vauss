"""Minimal Transformer Engine stand-in for Cosmos-Predict2 text2image inference/fine-tuning on sm_120.

Only what cosmos_predict2.models.text2image_dit needs: pytorch.RMSNorm, pytorch.attention.DotProductAttention
(no mask, bshd) and pytorch.attention.apply_rotary_pos_emb (non-interleaved). Math follows TE's unfused paths.
"""
__version__ = "0.0+shim"
from . import pytorch  # noqa: F401
