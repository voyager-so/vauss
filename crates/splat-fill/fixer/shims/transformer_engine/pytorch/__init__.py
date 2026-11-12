import torch
import torch.nn as nn
import torch.nn.functional as F


class _DropExtraState:
    """TE modules save an fp8 `_extra_state` entry; accept and ignore it so strict loads of TE checkpoints work."""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(prefix + "_extra_state", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class RMSNorm(_DropExtraState, nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, sequence_parallel=None, params_dtype=None,
                 zero_centered_gamma=False, device=None, **kw):
        super().__init__()
        n = normalized_shape if isinstance(normalized_shape, int) else int(normalized_shape[-1])
        self.n, self.eps, self.zero_centered_gamma = n, eps, zero_centered_gamma
        self.weight = nn.Parameter(torch.empty(n, device=device, dtype=params_dtype or torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight.device.type == "meta":
            return
        with torch.no_grad():
            self.weight.fill_(0.0 if self.zero_centered_gamma else 1.0)

    def forward(self, x):
        w = self.weight.float() + (1.0 if self.zero_centered_gamma else 0.0)
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w
        return y.to(x.dtype)


from . import attention  # noqa: E402,F401
