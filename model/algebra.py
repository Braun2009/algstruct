"""Algebraic structure stream in a language-model-shaped backbone.

Follows `docs/v5/MODEL_MAP.md` exactly; when code and that page disagree, the
page wins.

Per layer that carries the structural stream, and per token:

    d_t = W_d h_t                                    semantic -> structure (a source)
    da  = -grad V(a) - D(a) * sym(a) - Omega(a) * skew(a) + d_t
    a   = a + dt * da                                encoding  (a in gl(k))
    A   = Cayley(skew a) @ P(a)                      ||A|| <= 1 by construction
    Z   = A @ Z                                      computation: composition
    h   = h + W_o vec(Z)                             coupling: read the composition back

No injection into Z: content reaches the structure only by changing the law,
which is the point (structure is the operator, not a state that content is
added to).

Analytic potential gradient (checked against autograd to 3.6e-15):

    V(a)  = v1 Tr a + v2 Tr a^2 + v3 ||a||^2 + v4 ||[a, a^T]||^2
    grad  = v1 I + 2 v2 a^T + 2 v3 a + 4 v4 [C, a],   C = a a^T - a^T a

Arms
    full       as above
    nocompose  A = I (Z stays the identity): composition removed
    nodrive    d_t = 0: the semantic stream cannot reach the law
    off        no structural stream (plain transformer baseline)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .block import LocalBlock

ARMS = ('full', 'nocompose', 'nodrive', 'off')


@dataclass
class AlgebraConfig:
    width: int = 192
    layers: int = 6
    k: int = 3
    window: int = 8
    state_layers: tuple = (1, 3)
    arm: str = 'full'
    dt: float = .1
    step_limit: float = .5      # trust region on |da| per token
    skew_limit: float = 1.2     # keeps Cayley away from the singular point


def cayley(a: Tensor) -> Tensor:
    """R = (I - a/2)^{-1} (I + a/2) for skew a: exactly orthogonal."""
    k = a.shape[-1]
    eye = torch.eye(k, dtype=a.dtype, device=a.device)
    half = .5 * a
    return torch.linalg.solve(eye - half, eye + half)


class AlgebraState(nn.Module):
    """Encoding, computation and readout of the structural stream."""

    def __init__(self, cfg: AlgebraConfig):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k
        # 8 shared generator coefficients. v2, v3, v4 must be >= 0 so that V is
        # coercive (a cubic invariant such as det a would be unbounded below).
        self.raw_v = nn.Parameter(torch.tensor([0., .5, .5, .5]))   # v1, v2, v3, v4
        self.d = nn.Parameter(torch.tensor([1., 0.]))               # d0, d1
        self.omega = nn.Parameter(torch.tensor([0., 0.]))           # w0, w1
        # small random, not zero: with a zero drive every token would get the
        # same a, so the stream would start token-independent and the
        # "structure is not a constant" check could not be evaluated at all
        self.to_drive = nn.Linear(cfg.width, cfg.k * cfg.k)
        nn.init.normal_(self.to_drive.weight, std=1e-3)
        nn.init.zeros_(self.to_drive.bias)
        self.out = nn.Linear(cfg.k * cfg.k, cfg.width, bias=False)

    # ── the generator ──────────────────────────────────────────────
    def coeffs(self):
        v = torch.stack((self.raw_v[0], F.softplus(self.raw_v[1]),
                         F.softplus(self.raw_v[2]), F.softplus(self.raw_v[3])))
        return v, self.d[0], self.d[1], self.omega[0], self.omega[1]

    def potential(self, a: Tensor, v: Tensor) -> Tensor:
        a2 = a @ a
        c = a @ a.mT - a.mT @ a
        return (v[0] * a.diagonal(dim1=-2, dim2=-1).sum(-1)
                + v[1] * a2.diagonal(dim1=-2, dim2=-1).sum(-1)
                + v[2] * a.square().sum((-1, -2))
                + v[3] * c.square().sum((-1, -2)))

    def grad_potential(self, a: Tensor, v: Tensor) -> Tensor:
        """Analytic; verified against autograd in tests_v5/test_algebra_model.py."""
        c = a @ a.mT - a.mT @ a
        comm = c @ a - a @ c
        eye = torch.eye(self.k, dtype=a.dtype, device=a.device)
        return (v[0] * eye + 2 * v[1] * a.mT + 2 * v[2] * a + 4 * v[3] * comm)

    def flow(self, a: Tensor, drive: Tensor) -> Tensor:
        v, d0, d1, w0, w1 = self.coeffs()
        r2 = a.square().sum((-1, -2), keepdim=True)
        dissip = d0 + d1 * r2
        gyro = w0 + w1 * r2
        sym = .5 * (a + a.mT)
        skew = .5 * (a - a.mT)
        return -self.grad_potential(a, v) - dissip * sym - gyro * skew + drive

    def decompose(self, a: Tensor):
        """A = R P with ||R||=1 and ||P||<=1, exactly.

        P = exp(sym - shift I) with shift = ||sym||_F.  Since the spectral norm
        of a symmetric matrix is at most its Frobenius norm, shift >= lmax and
        every eigenvalue of P lies in (0, 1].

        Why not use lmax itself: eigh fails on degenerate spectra and its
        backward pass returns NaN when eigenvalues repeat -- and a repeated
        spectrum is exactly the case sym -> 0, i.e. the start of training.
        The cost of the Frobenius shift is a conservative over-contraction when
        the spectrum is isotropic; it vanishes as sym -> 0, so a direction the
        model wants to preserve is preserved by driving sym there.
        """
        skew = .5 * (a - a.mT)
        sym = .5 * (a + a.mT)
        # Cayley is singular where I - skew/2 degenerates. Rescaling the
        # generator is harmless: the stream composes over time, so a large
        # per-step rotation can be reached in several steps.
        norm = skew.square().sum((-1, -2), keepdim=True).sqrt()
        skew = skew * (norm.clamp(max=self.cfg.skew_limit) / norm.clamp(min=1e-12))
        r = cayley(skew)
        shift = sym.square().sum((-1, -2), keepdim=True).sqrt()
        eye = torch.eye(self.k, dtype=a.dtype, device=a.device)
        p = torch.matrix_exp(sym - shift * eye)
        return r, p

    # ── the stream ─────────────────────────────────────────────────
    def _prefix_product(self, mats: Tensor) -> Tensor:
        """Inclusive prefix product of (b,t,k,k) matrices, in log2(t) matmuls.

        Z_t = A_t Z_{t-1} with Z_0 = I, so Z_t equals the prefix product
        A_t ... A_1. Matrices do not commute, which is why a plain cumprod is
        wrong and a doubling scan is used instead: after step d the entry at t
        holds the product over the last 2^d factors.
        """
        b, t, k, _ = mats.shape
        eye = torch.eye(k, dtype=mats.dtype, device=mats.device)
        out, step = mats, 1
        while step < t:
            shifted = eye.expand(b, t, k, k).clone()
            shifted[:, step:] = out[:, :-step]
            out = shifted @ out
            step *= 2
        return out

    def forward(self, h: Tensor, stats: dict | None = None):
        # the Cayley solve and matrix_exp are done in fp32 regardless of the
        # surrounding autocast: bf16 there costs accuracy and buys nothing
        dtype = h.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            h32 = h.float()
            out = self._stream(h32, stats)
        return out.to(dtype)

    def _stream(self, h: Tensor, stats: dict | None = None):
        b, t, _ = h.shape
        eye = torch.eye(self.k, dtype=h.dtype, device=h.device).expand(b, -1, -1).contiguous()
        a = torch.zeros(b, self.k, self.k, dtype=h.dtype, device=h.device)
        a_traj = []
        # Only the cheap elementwise part is sequential. matrix_exp and the
        # Cayley solve are done once, batched over all positions, because a_t
        # itself does not depend on Z.
        for i in range(t):
            drive = (torch.zeros_like(a) if self.cfg.arm == 'nodrive'
                     else self.to_drive(h[:, i]).reshape(b, self.k, self.k))
            da = self.flow(a, drive)
            da = da / (1. + da.norm(dim=(-2, -1), keepdim=True) / self.cfg.step_limit)
            a = a + self.cfg.dt * da
            a_traj.append(a)
        a_seq = torch.stack(a_traj, 1)                       # (b,t,k,k)
        r, p = self.decompose(a_seq)
        mats = (r @ p) if self.cfg.arm != 'nocompose' else eye[:, None].expand_as(a_seq)
        z = self._prefix_product(mats) @ eye[:, None]        # (b,t,k,k)
        if stats is not None:
            stats['a_var'] = float(a_seq.detach().var(dim=1).mean())
            zf = torch.linalg.matrix_norm(z.detach(), dim=(-2, -1))
            stats['z_norm'] = float(zf.mean())
            stats['z_last'] = float(zf[:, -1].mean())
            stats['mat_norm'] = float(torch.linalg.matrix_norm(mats.detach(), dim=(-2, -1)).max(dim=-1).values.mean())
        return h + self.out(z.reshape(b, t, -1))


class AlgebraModel(nn.Module):
    """Backbone + structural stream. `vocab` and `n_fields` describe the input."""

    def __init__(self, cfg: AlgebraConfig, vocab: int, n_fields: int = 4):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.ModuleList(nn.Embedding(vocab, cfg.width) for _ in range(n_fields))
        self.blocks = nn.ModuleList(LocalBlock(cfg.width, cfg.window) for _ in range(cfg.layers))
        self.state = nn.ModuleDict({str(i): AlgebraState(cfg) for i in cfg.state_layers
                                    if cfg.arm != 'off'})
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, vocab)

    def forward(self, events: Tensor, stats: dict | None = None):
        x = sum(emb(events[..., i]) for i, emb in enumerate(self.enc))
        for i, blk in enumerate(self.blocks):
            x = x + blk(x)
            if str(i) in self.state:
                x = self.state[str(i)](x, stats)
        return self.head(self.norm(x))

    def generators(self):
        return {name: mod.coeffs() for name, mod in self.state.items()}
