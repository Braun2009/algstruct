"""The semantic side: causal attention with rotary positions + MLP.

Deliberately separate from the structural stream. This block computes
commutative second-order statistics (q.k) and averages over positions; the
structural stream never goes through it (see docs/DESIGN.md).

RoPE is here because the two streams need different notions of position.
The structural stream already carries order intrinsically -- `Z_t = A_t Z_{t-1}`
is non-commutative, so "A then B" and "B then A" are different matrices with no
positional input at all. The semantic stream has no such thing: attention on its
own is permutation-equivariant over the keys, so it cannot tell position 3 from
position 300. Giving it a rotation by angle theta*t makes `q_i . k_j` depend on
the difference `j - i`, which is the semantic side's own way of encoding order --
a second, independent channel, commuting in type with the structural one.

Rotary is applied to q and k only, never to v: v is what gets averaged, and
averaging a rotated value would destroy it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RoPE(nn.Module):
    """Rotate each (q,k) pair of channels by an angle proportional to position.

    Splitting the head into pairs and rotating pair j by theta_j * t is what
    makes the dot product of two rotated vectors depend only on t_q - t_k.
    The frequencies are geometric, as in the original construction:

        theta_j = base^(-2j/d),   j = 0 .. d/2-1
    """

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError('rotary needs an even head dimension')
        self.dim = dim
        inv = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        self.register_buffer('inv_freq', inv, persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """x: (b, heads, t, dim) -> same shape, positions rotated."""
        t = x.shape[-2]
        pos = torch.arange(offset, offset + t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)                 # (t, dim/2)
        cos, sin = freqs.cos(), freqs.sin()
        # (1,1,t,dim/2) so it broadcasts over batch and heads
        cos, sin = cos[None, None], sin[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin,
                            x1 * sin + x2 * cos), dim=-1).flatten(-2)


class LocalBlock(nn.Module):
    def __init__(self, width: int, window: int | None = None, heads: int = 4,
                 rope_base: float = 10_000.0):
        super().__init__()
        if width % 64:
            raise ValueError('width must be a multiple of 64')
        self.window = window
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))
        self.rope = RoPE(64, rope_base)

    def _mask(self, t: int, device) -> Tensor:
        """Causal. With `window` set, also banded.

        The default is full causal attention: the benchmark scatters program
        events over the whole sequence and asks at the last position, so a
        narrow band cannot answer at all -- a window of 8 leaves every task
        except the purely local one at chance, which is exactly what was
        measured before this was fixed.
        """
        i = torch.arange(t, device=device)
        mask = i[:, None] >= i[None, :]
        if self.window is not None:
            mask = mask & ((i[:, None] - i[None, :]) < max(self.window, 1))
        return mask[None, None]

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, t, 3, d // 64, 64).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q), self.rope(k)
        msg = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(t, x.device))
        x = x + self.out(msg.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.n2(x))
