"""Algebraic structure stream in a language-model-shaped backbone.

Follows `docs/MODEL_MAP.md` exactly; when code and that page disagree, the
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

    V(a)  = v2 Tr a^2 + v3 ||a||^2 + v4 ||[a, a^T]||^2
    grad  = 2 v2 a^T + 2 v3 a + 4 v4 [C, a],   C = a a^T - a^T a

Arms
    full       as above
    nocompose  Z = A: per-token transform kept, accumulation removed
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
        # 7 shared generator coefficients: 3 for V, 2 for D, 2 for Omega.
        # v2, v3, v4 are softplus-constrained to stay positive so that V is
        # coercive (a cubic invariant such as det a would be unbounded below).
        # Tr a is deliberately absent: a is projected to traceless every step, so
        # that term would be a constant with zero gradient -- a dead parameter.
        self.raw_v = nn.Parameter(torch.tensor([.5, .5, .5]))       # v2, v3, v4
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
        v = torch.stack((F.softplus(self.raw_v[0]), F.softplus(self.raw_v[1]),
                         F.softplus(self.raw_v[2])))
        return v, self.d[0], self.d[1], self.omega[0], self.omega[1]

    def potential(self, a: Tensor, v: Tensor) -> Tensor:
        a2 = a @ a
        c = a @ a.mT - a.mT @ a
        return (v[0] * a2.diagonal(dim1=-2, dim2=-1).sum(-1)
                + v[1] * a.square().sum((-1, -2))
                + v[2] * c.square().sum((-1, -2)))

    def grad_potential(self, a: Tensor, v: Tensor) -> Tensor:
        """Analytic; agrees with autograd to 3.6e-15."""
        c = a @ a.mT - a.mT @ a
        comm = c @ a - a @ c
        return 2 * v[0] * a.mT + 2 * v[1] * a + 4 * v[2] * comm

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

        P = exp(sym - lmax I), so the spectrum of P lies in (0, 1] with the
        top eigenvalue exactly 1. The shift must be lmax itself, not an upper
        bound on it: with shift = ||sym||_F an isotropic contraction (sym = cI,
        c < 0) gives P = exp(-3|c| I), and 256 compositions drive Z to exactly
        zero, which silently disconnects the readout. Measured on the first full
        run: z_last = 0.0 at 256 tokens.

        lmax is computed in float64 from a detached copy and the result is
        detached: the shift is a normaliser, and eigh's backward pass returns
        NaN when eigenvalues repeat (exactly the case sym -> 0 at init). If eigh
        fails, fall back to the Frobenius bound, which is conservative but safe.
        """
        skew = .5 * (a - a.mT)
        sym = .5 * (a + a.mT)
        # Cayley is singular where I - skew/2 degenerates. Rescaling the
        # generator is harmless: the stream composes over time.
        norm = skew.square().sum((-1, -2), keepdim=True).sqrt()
        skew = skew * (norm.clamp(max=self.cfg.skew_limit) / norm.clamp(min=1e-12))
        r = cayley(skew)
        eye = torch.eye(self.k, dtype=a.dtype, device=a.device)
        with torch.no_grad():
            s64 = sym.detach().double()
            try:
                shift = torch.linalg.eigvalsh(s64)[..., -1:].unsqueeze(-1)
            except Exception:
                shift = s64.square().sum((-1, -2), keepdim=True).sqrt()
        p = torch.matrix_exp(sym - shift.to(sym.dtype) * eye)
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
            out = out @ shifted
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
        # W_d has no recurrence: apply it to the whole sequence in one GEMM
        # instead of one small GEMM per position
        if self.cfg.arm == 'nodrive':
            drives = torch.zeros(b, t, self.k, self.k, dtype=h.dtype, device=h.device)
        else:
            drives = self.to_drive(h).reshape(b, t, self.k, self.k)
        for i in range(t):
            drive = drives[:, i]
            da = self.flow(a, drive)
            da = da / (1. + da.norm(dim=(-2, -1), keepdim=True) / self.cfg.step_limit)
            a = a + self.cfg.dt * da
            # Gauge fixing. a -> a + cI leaves A unchanged (sym and lmax shift
            # together and skew is untouched), but the drive W_d h is not
            # traceless, so a drifts along I without bound and matrix_exp
            # overflows in fp32. Projecting a to traceless each step removes the
            # drift and changes nothing about the transform.
            tr = a.diagonal(dim1=-2, dim2=-1).sum(-1) / self.k
            a = a - tr[..., None, None] * torch.eye(self.k, dtype=a.dtype,
                                                   device=a.device)
            a_traj.append(a)
        a_seq = torch.stack(a_traj, 1)                       # (b,t,k,k)
        r, p = self.decompose(a_seq)
        mats = r @ p
        if self.cfg.arm == 'nocompose':
            # no accumulation, but the per-token transform is kept: this isolates
            # "composition along time" from "a transform exists". Replacing A by I
            # instead would make the readout a constant and test nothing.
            z = mats
        else:
            z = self._prefix_product(mats) @ eye[:, None]    # (b,t,k,k)
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
