"""Algebraic structure stream — v6_hom (pure homomorphism, no RNN shortcut).

Why v6_hom exists (empirical finding from v5_tau):
  In v5_tau, `a_t = _advance(a_{t-1}, drive_t)` carried a 64-dim nonlinear RNN
  state across time *in addition* to the matrix product Z_t = A_t Z_{t-1}.
  On S3 (L_train=16), `nocompose` (NO prefix product!) scored 0.995 in-dist
  by memorizing inside the a_t RNN, and then collapsed to 0.236 at 2x length.
  Even `full` leaned on the a_t shortcut (z_ratio = 0.10, a = 4.95) and failed
  to extrapolate to L = 64.

Fix in v6_hom:
  1. Per-token algebraic encoding: the global 7-coefficient law (V, D, Omega)
     shapes each token's local generator a_t in `n_relax` parallel ODE steps
     from a_0 = 0. There is NO cross-token recurrence in a_t -- all memory
     across time MUST live in the non-commutative matrix product Z_t.
     Bonus: eliminates the T-step sequential Python loop (5x+ faster).
  2. Proper-time scaling on BOTH factors:
         R_t = exp(tau_t * skew(a_t))              in SO(k), ||R_t||_2 = 1
         P_t = exp(tau_t * (sym(a_t) - lmax I))    spectrum in (0, 1]
     When tau_t -> 0 (filler/NOISE token), BOTH R_t = I and P_t = I, so
     A_t = I exactly (strict invariance to no-ops).
  3. Two composition modes (CLI selectable):
         rot      Z_t = P_t @ prefix_product(R)     isometric SO(k) product
         full_ap  Z_t = prefix_product(R @ P)       monoid with partial inverse
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .action import rho_split
from .block import LocalBlock

ARMS = ('full', 'nocompose', 'nodrive', 'off')


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def cayley(a: Tensor) -> Tensor:
    """Kept for API parity with mainline."""
    k = a.shape[-1]
    eye = torch.eye(k, dtype=a.dtype, device=a.device)
    half = .5 * a
    return torch.linalg.solve(eye - half, eye + half)


def _relax_step(a: Tensor, drive: Tensor, v: Tensor, d0: Tensor, d1: Tensor,
                w0: Tensor, w1: Tensor, dt: float, step_limit: float,
                a_limit: float, eye: Tensor) -> Tensor:
    """One parallel relaxation step on (b, t, k, k) under the global law."""
    r2 = a.square().sum((-1, -2), keepdim=True)
    c = a @ a.mT - a.mT @ a
    grad = 2 * v[0] * a.mT + 2 * v[1] * a + 4 * v[2] * (c @ a - a @ c)
    dissip = d0 + d1 * r2
    gyro = w0 + w1 * r2
    da = -grad - dissip * (.5 * (a + a.mT)) - gyro * (.5 * (a - a.mT)) + drive
    da = da / (1. + da.norm(dim=(-2, -1), keepdim=True) / step_limit)
    a = a + dt * da
    tr = a.diagonal(dim1=-2, dim2=-1).sum(-1) / a.shape[-1]
    a = a - tr[..., None, None] * eye
    n = a.norm(dim=(-2, -1), keepdim=True)
    return a * (a_limit / n.clamp(min=a_limit))


@dataclass
class AlgebraConfig:
    width: int = 192
    layers: int = 6
    k: int = 8
    q: int = 64
    drive_scale: float = 1.0
    window: int | None = None
    state_layers: tuple = (1, 3)
    arm: str = 'full'
    compose_mode: str = 'rot'   # 'rot' (P_t @ prod R) or 'full_ap' (prod(R @ P))
    n_relax: int = 2            # parallel law-relaxation steps per token
    dt: float = 1.0             # maximum proper-time step per token
    tau_bias: float = 0.0       # initial logit for tau; sigmoid(0) = 0.5
    memory: float = 2.0         # sets initial stiffness v3 = 1/(2 * memory * 0.5)
    out_init: float = 3e-3
    step_limit: float = 6.0
    a_limit: float = 5.0        # covers sqrt(2)*pi = 4.44 (180 deg in one plane)
    skew_limit: float = 1.2     # unused by exp map; kept for config parity


class Generator(nn.Module):
    """The 7 coefficients of the law, shared by every structural layer."""

    def __init__(self, dt: float = 0.5, memory: float = 2.0):
        super().__init__()
        rate = 1.0 / max(memory * dt, 1e-3)     # = 2 * v3
        v3 = .5 * rate
        v2 = .5 * v3                             # guarantees v3 > v2
        v4 = .1 * v3
        self.raw_v = nn.Parameter(torch.tensor([_inv_softplus(v2),
                                                _inv_softplus(v3 - v2),
                                                _inv_softplus(v4)]))
        self.d = nn.Parameter(torch.tensor([0., 0.]))
        self.omega = nn.Parameter(torch.tensor([0., 0.]))

    def coeffs(self):
        v2 = F.softplus(self.raw_v[0])
        v = torch.stack((v2, v2 + F.softplus(self.raw_v[1]), F.softplus(self.raw_v[2])))
        return v, self.d[0], self.d[1], self.omega[0], self.omega[1]

    def values(self):
        v, d0, d1, w0, w1 = self.coeffs()
        d = v.detach()
        return {'v2': float(d[0]), 'v3': float(d[1]), 'v4': float(d[2]),
                'd0': float(d0.detach()), 'd1': float(d1.detach()),
                'w0': float(w0.detach()), 'w1': float(w1.detach())}


class AlgebraState(nn.Module):
    """Per-token algebraic encoding, prefix-product composition, and rho action."""

    def __init__(self, cfg: AlgebraConfig, gen: Generator):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k
        self.q = cfg.q
        self.gen = gen

        self.dn = nn.LayerNorm(cfg.width)
        self.to_drive = nn.Linear(cfg.width, cfg.k * cfg.k)
        nn.init.normal_(self.to_drive.weight,
                        std=cfg.drive_scale / (cfg.k * math.sqrt(cfg.width)))
        nn.init.zeros_(self.to_drive.bias)

        # proper-time step: one positive scalar per token in (0, dt).
        self.to_tau = nn.Linear(cfg.width, 1)
        nn.init.normal_(self.to_tau.weight, std=1e-2)
        nn.init.constant_(self.to_tau.bias, cfg.tau_bias)

        # interface: semantic <-> structural space
        self.to_s = nn.Linear(cfg.width, cfg.k * cfg.q, bias=False)
        self.out = nn.Linear(cfg.k * cfg.q, cfg.width, bias=False)
        if cfg.out_init == 0:
            nn.init.zeros_(self.out.weight)
        else:
            nn.init.normal_(self.out.weight, std=cfg.out_init)

    def coeffs(self):
        return self.gen.coeffs()

    def decompose(self, a: Tensor, tau: Tensor):
        """A_t = R_t P_t with ||R_t||_2 = 1 and ||P_t||_2 <= 1, exactly.

        When tau_t -> 0, both R_t -> I and P_t -> I, so A_t -> I (exact no-op).
        """
        skew = .5 * (a - a.mT)
        sym = .5 * (a + a.mT)
        r = torch.matrix_exp(tau * skew)
        eye = torch.eye(self.k, dtype=a.dtype, device=a.device)
        with torch.no_grad():
            s64 = sym.detach().double()
            try:
                shift = torch.linalg.eigvalsh(s64)[..., -1:].unsqueeze(-1)
            except Exception:
                shift = s64.square().sum((-1, -2), keepdim=True).sqrt()
        p = torch.matrix_exp(tau * (sym - shift.to(sym.dtype) * eye))
        return r, p

    def _prefix_product(self, mats: Tensor) -> Tensor:
        """Inclusive left prefix product A_t ... A_1 in ceil(log2(t)) matmuls."""
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
        dtype = h.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            out = self._stream(h.float(), stats)
        return out.to(dtype)

    def _stream(self, h: Tensor, stats: dict | None = None):
        b, t, _ = h.shape
        k, q = self.k, self.q
        if self.cfg.arm == 'nodrive':
            drives = torch.zeros(b, t, k, k, dtype=h.dtype, device=h.device)
            taus = torch.zeros(b, t, 1, 1, dtype=h.dtype, device=h.device)
        else:
            hn = self.dn(h)
            drives = self.to_drive(hn).reshape(b, t, k, k)
            taus = (self.cfg.dt * torch.sigmoid(self.to_tau(hn))).unsqueeze(-1)

        # Parallel per-token relaxation under the global 7-parameter law.
        # Starts from a = 0 at every position: zero cross-token RNN shortcut.
        v, d0, d1, w0, w1 = self.coeffs()
        eye_a = torch.eye(k, dtype=h.dtype, device=h.device)
        a_seq = torch.zeros_like(drives)
        dt_relax = 0.5
        for _ in range(self.cfg.n_relax):
            a_seq = _relax_step(a_seq, drives, v, d0, d1, w0, w1,
                                dt_relax, self.cfg.step_limit, self.cfg.a_limit, eye_a)

        r, p = self.decompose(a_seq, taus)
        if self.cfg.arm == 'nocompose':
            z = r @ p
        elif self.cfg.compose_mode == 'full_ap':
            z = self._prefix_product(r @ p)
        else:
            z = p @ self._prefix_product(r)

        if stats is not None:
            zf = torch.linalg.matrix_norm(z.detach(), dim=(-2, -1))
            a_d = a_seq.detach()
            a_var = float(a_d.var(dim=1).mean())
            a_ms = float(a_d.square().sum((-1, -2)).mean())
            stats['a_var'] = a_var
            stats['a_scale'] = a_ms ** .5
            stats['a_rel'] = a_var / a_ms if a_ms > 0 else 0.0
            stats['z_norm'] = float(zf.mean())
            stats['z_last'] = float(zf[:, -1].mean())
            stats['z_ratio'] = float(zf[:, -1].mean()) / (k ** .5)
            stats['tau_mean'] = float(taus.detach().mean())
            stats['tau_std'] = float(taus.detach().std())
            stats['R_norm_last'] = float(torch.linalg.matrix_norm(
                self._prefix_product(r.detach())[:, -1], dim=(-2, -1)).mean())

        s = self.to_s(h).reshape(b, t, k, q)
        return h + self.out(rho_split(z, s).reshape(b, t, k * q))


class AlgebraModel(nn.Module):
    """Backbone + structural stream."""

    def __init__(self, cfg: AlgebraConfig, vocab: int, n_fields: int = 4):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.ModuleList(nn.Embedding(vocab, cfg.width) for _ in range(n_fields))
        self.blocks = nn.ModuleList(LocalBlock(cfg.width, cfg.window) for _ in range(cfg.layers))
        self.gen = Generator(0.5, cfg.memory)
        self.state = nn.ModuleDict({str(i): AlgebraState(cfg, self.gen)
                                    for i in cfg.state_layers if cfg.arm != 'off'})
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, vocab)

    def forward(self, events: Tensor, stats: dict | None = None):
        x = sum(emb(events[..., i]) for i, emb in enumerate(self.enc))
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if str(i) in self.state:
                x = self.state[str(i)](x, stats)
        return self.head(self.norm(x))

    def generators(self):
        return self.gen.values()
