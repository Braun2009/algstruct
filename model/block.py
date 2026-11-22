"""The semantic side: local causal attention + MLP.

Deliberately separate from the structural stream. This block computes
commutative second-order statistics (q.k) and averages over positions; the
structural stream never goes through it (see docs/MODEL_MAP.md, section 2).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LocalBlock(nn.Module):
    def __init__(self, width: int, window: int = 8, heads: int = 4):
        super().__init__()
        if width % 64:
            raise ValueError('width must be a multiple of 64')
        self.window = window
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))

    def _mask(self, t: int, device) -> Tensor:
        i = torch.arange(t, device=device)
        causal = i[:, None] >= i[None, :]
        band = (i[:, None] - i[None, :]) < max(self.window, 1)
        return (causal & band)[None, None]

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, t, 3, d // 64, 64).permute(2, 0, 3, 1, 4)
        msg = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(t, x.device))
        x = x + self.out(msg.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.n2(x))
