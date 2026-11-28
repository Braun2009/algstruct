"""S3 word problem with per-position supervision and length extrapolation.

Trains on length T_train (default 16) and evaluates on T = 16, 32, 64, 128.
Uses the exact 4-field event interface of AlgebraModel with zero changes to
`model/`.

Why this probe exists:
  1. Final-token-only supervision at T=128 has 1/128 supervision density and
     can trap every arm at the uniform saddle point ln(6) = 1.792.
  2. Per-position supervision tests whether the algebraic stream can actually
     learn a non-abelian group homomorphism (S3, 6 classes; abelian parity alone
     caps at 0.333).
  3. Length extrapolation (16 -> 32 -> 64 -> 128) directly separates a
     full-attention transformer (replays prefix, breaks out of distribution)
     from an algebraic composition stream (carries the group element in Z_t).
  4. Comparing `full` vs `nocompose` under window=1 isolates matrix composition
     Z_t = A_t Z_{t-1} from the a_t ODE alone.

Run from the root of any variant (or mainline):
    python -m probes.s3_extrap --arms off_fullattn,off_w1,nocompose,full
"""
from __future__ import annotations

import argparse
import itertools
import math
import time

import torch
from torch.nn import functional as F

from model import AlgebraConfig, AlgebraModel

PERMS = tuple(itertools.permutations(range(3)))
# 0 = identity (NOISE), 1 = (0,2,1) [order 2, odd], 2 = (1,2,0) [order 3, even]
GEN_PERMS = (PERMS[0], PERMS[1], PERMS[3])

# Multiplication table: MUL[state, token] -> next_state (left action: g_t o s)
def _compose(g, s):
    return tuple(s[g[i]] for i in range(3))

MUL = torch.tensor(
    [[PERMS.index(_compose(GEN_PERMS[tok], PERMS[st])) for tok in range(3)]
     for st in range(6)],
    dtype=torch.long,
)
PARITY = torch.tensor(
    [sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j]) % 2
     for p in PERMS],
    dtype=torch.long,
)


def make_batch(batch: int, length: int, noise: float, seed: int, device):
    g = torch.Generator().manual_seed(seed)
    if noise > 0:
        is_noise = torch.rand(batch, length, generator=g) < noise
        ops = torch.randint(1, 3, (batch, length), generator=g)
        ops = torch.where(is_noise, torch.zeros_like(ops), ops)
    else:
        ops = torch.randint(1, 3, (batch, length), generator=g)
    states = torch.zeros(batch, length, dtype=torch.long)
    cur = torch.zeros(batch, dtype=torch.long)
    for t in range(length):
        cur = MUL[cur, ops[:, t]]
        states[:, t] = cur
    ev = torch.zeros(batch, length, 4, dtype=torch.long)
    ev[..., 0] = ops
    return ev.to(device), states.to(device)


@torch.no_grad()
def eval_lengths(model, lengths, noise, device, n=512):
    model.eval()
    lut = PARITY.to(device)
    out = {}
    for L in lengths:
        ev, lab = make_batch(n, L, noise, 9000 + L, device)
        st = {}
        logits = model(ev, st)
        pred = logits.argmax(-1)
        acc_all = float((pred == lab).float().mean())
        acc_last = float((pred[:, -1] == lab[:, -1]).float().mean())
        m = pred[:, -1] < 6
        par_last = float(torch.where(
            m, lut[pred[:, -1].clamp(max=5)] == lut[lab[:, -1]],
            torch.zeros_like(m)).float().mean())
        out[L] = (acc_all, acc_last, par_last, st)
    model.train()
    return out


def build_cfg(arm_spec, a):
    if arm_spec == 'off_fullattn':
        arm, window = 'off', None
    elif arm_spec == 'off_w1':
        arm, window = 'off', 1
    else:
        arm, window = arm_spec, (a.window or None)
    kw = dict(
        width=a.width, layers=a.layers, k=a.k, q=a.q,
        state_layers=tuple(int(x) for x in a.state_layers.split(',')),
        arm=arm, window=window, dt=a.dt, step_limit=a.step_limit,
        a_limit=a.a_limit, drive_scale=a.drive_scale,
    )
    # pass optional variant fields only if supported by the active AlgebraConfig
    fields = set(AlgebraConfig.__dataclass_fields__)
    for k_opt, v_opt in (('memory', a.memory), ('out_init', a.out_init),
                         ('tau_bias', a.tau_bias), ('compose_mode', a.compose_mode)):
        if k_opt in fields:
            kw[k_opt] = v_opt
    return AlgebraConfig(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='off_fullattn,off_w1,nocompose,full')
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--q', type=int, default=32)
    ap.add_argument('--state-layers', default='1,3')
    ap.add_argument('--window', type=int, default=1)
    ap.add_argument('--train-len', type=int, default=16)
    ap.add_argument('--eval-lens', default='16,32,64,128')
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--memory', type=float, default=2.0)
    ap.add_argument('--out-init', type=float, default=3e-3)
    ap.add_argument('--dt', type=float, default=1.0)
    ap.add_argument('--tau-bias', type=float, default=-1.0)
    ap.add_argument('--compose-mode', default='rot')
    ap.add_argument('--step-limit', type=float, default=6.0)
    ap.add_argument('--a-limit', type=float, default=5.0)
    ap.add_argument('--drive-scale', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    eval_lens = [int(x) for x in a.eval_lens.split(',')]
    print(f'S3 extrapolation: train_len={a.train_len} noise={a.noise} steps={a.steps} '
          f'w={a.width} L={a.layers} k={a.k} q={a.q} win={a.window} '
          f'dt={a.dt} mem={a.memory} a_lim={a.a_limit}', flush=True)
    hdr = f'{"arm":14s} {"loss":>6s} {"secs":>5s}  ' + '  '.join(
        f'L={L:<3d}(all/last/par)' for L in eval_lens)
    print(hdr, flush=True)

    for arm_spec in a.arms.split(','):
        torch.manual_seed(a.seed)
        cfg = build_cfg(arm_spec, a)
        model = AlgebraModel(cfg, vocab=64).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: .5 * (1 + math.cos(math.pi * min(1., s / a.steps))))
        t0 = time.time()
        for step in range(1, a.steps + 1):
            ev, lab = make_batch(a.batch, a.train_len, a.noise, 1000 + step, dev)
            logits = model(ev)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
        res = eval_lengths(model, eval_lens, a.noise, dev)
        cols = '  '.join(f'{res[L][0]:.3f}/{res[L][1]:.3f}/{res[L][2]:.3f}'
                         for L in eval_lens)
        print(f'{arm_spec:14s} {float(loss.detach()):6.3f} {time.time() - t0:5.0f}  {cols}',
              flush=True)
        st = res[eval_lens[-1]][3]
        if st:
            g = model.generators()
            tau_s = (f' tau={st["tau_mean"]:.2f}±{st["tau_std"]:.2f}'
                     if 'tau_mean' in st else '')
            print(f'{"":14s} law v2={g["v2"]:.4f} v3={g["v3"]:.4f} '
                  f'z={st.get("z_ratio", 0):.3f} a={st.get("a_scale", 0):.3f}{tau_s}',
                  flush=True)


if __name__ == '__main__':
    main()
