"""Non-abelian group word-problem & length-extrapolation probe: S3, S4, A5.

Why these three groups:
  S3  order  6   smallest non-abelian group (solvable, Z2 abelianization)
  S4  order 24   non-abelian (solvable, 4! states)
  A5  order 60   smallest non-abelian SIMPLE group (Barrington's theorem:
                 word multiplication in A5 is NC^1-complete; no nontrivial
                 abelian quotient exists, so commutative aggregation cannot
                 even capture parity).

Trains on length L_train (default 16) with per-position supervision and
evaluates up to L = 1024 (64x extrapolation). Also tests the hybrid
`full_fullattn` arm AND runs an immediate `out = 0` ablation on it to check
whether the structural stream is load-bearing at ID and OOD lengths.

    python -m probes.group_extrap --group A5 --arms off_fullattn,nocompose,full,full_fullattn
"""
from __future__ import annotations

import argparse
import itertools
import math
import time

import torch
from torch.nn import functional as F

from model import AlgebraConfig, AlgebraModel


def _compose(g, s):
    return tuple(s[g[i]] for i in range(len(s)))


def _is_even(p):
    inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return (inv % 2) == 0


def build_group(name: str):
    if name == 'S3':
        elems = list(itertools.permutations(range(3)))
        gens = [elems[0], (0, 2, 1), (1, 2, 0)]                      # e, (12), (012)
    elif name == 'S4':
        elems = list(itertools.permutations(range(4)))
        gens = [elems[0], (1, 0, 2, 3), (1, 2, 3, 0)]                # e, (01), (0123)
    elif name == 'A5':
        elems = [p for p in itertools.permutations(range(5)) if _is_even(p)]
        # (012) and (01234) generate A5 (order 60, simple)
        gens = [elems[0], (1, 2, 0, 3, 4), (1, 2, 3, 4, 0)]
    else:
        raise ValueError(f'unknown group {name}')
    idx = {p: i for i, p in enumerate(elems)}
    # verify closure and generation
    seen = {elems[0]}
    frontier = [elems[0]]
    while frontier:
        cur = frontier.pop()
        for g in gens[1:]:
            nxt = _compose(g, cur)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert len(seen) == len(elems), f'generators only cover {len(seen)}/{len(elems)}'
    mul = torch.tensor(
        [[idx[_compose(g, s)] for g in gens] for s in elems], dtype=torch.long)
    return len(elems), mul


def make_batch(mul, batch: int, length: int, noise: float, seed: int, device):
    n_gens = mul.shape[1] - 1
    g = torch.Generator().manual_seed(seed)
    ops = torch.randint(1, 1 + n_gens, (batch, length), generator=g)
    if noise > 0:
        ops = torch.where(torch.rand(batch, length, generator=g) < noise,
                          torch.zeros_like(ops), ops)
    states = torch.zeros(batch, length, dtype=torch.long)
    cur = torch.zeros(batch, dtype=torch.long)
    for t in range(length):
        cur = mul[cur, ops[:, t]]
        states[:, t] = cur
    ev = torch.zeros(batch, length, 4, dtype=torch.long)
    ev[..., 0] = ops
    return ev.to(device), states.to(device)


@torch.no_grad()
def eval_lengths(model, mul, lengths, noise, device, n=256):
    model.eval()
    out = {}
    for L in lengths:
        ev, lab = make_batch(mul, n, L, noise, 9000 + L, device)
        st = {}
        logits = model(ev, st)
        pred = logits.argmax(-1)
        out[L] = (float((pred == lab).float().mean()),
                  float((pred[:, -1] == lab[:, -1]).float().mean()),
                  st)
    model.train()
    return out


def build_cfg(arm_spec, a):
    if arm_spec == 'off_fullattn':
        arm, window = 'off', None
    elif arm_spec == 'full_fullattn':
        arm, window = 'full', None
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
    fields = set(AlgebraConfig.__dataclass_fields__)
    for k_opt, v_opt in (('memory', a.memory), ('out_init', a.out_init),
                         ('tau_bias', a.tau_bias), ('compose_mode', a.compose_mode)):
        if k_opt in fields:
            kw[k_opt] = v_opt
    return AlgebraConfig(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='S4', choices=('S3', 'S4', 'A5'))
    ap.add_argument('--arms', default='off_fullattn,nocompose,full,full_fullattn')
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--q', type=int, default=32)
    ap.add_argument('--state-layers', default='1,3')
    ap.add_argument('--window', type=int, default=1)
    ap.add_argument('--train-len', type=int, default=16)
    ap.add_argument('--eval-lens', default='16,32,64,128,256,512,1024')
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--memory', type=float, default=2.0)
    ap.add_argument('--out-init', type=float, default=3e-3)
    ap.add_argument('--dt', type=float, default=1.0)
    ap.add_argument('--tau-bias', type=float, default=0.0)
    ap.add_argument('--compose-mode', default='rot')
    ap.add_argument('--step-limit', type=float, default=6.0)
    ap.add_argument('--a-limit', type=float, default=5.0)
    ap.add_argument('--drive-scale', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    order, mul = build_group(a.group)
    eval_lens = [int(x) for x in a.eval_lens.split(',')]
    print(f'Group {a.group} (order={order}, chance={1/order:.4f})  train_len={a.train_len} '
          f'noise={a.noise} steps={a.steps} w={a.width} L={a.layers} k={a.k} q={a.q}',
          flush=True)
    hdr = f'{"arm":16s} {"loss":>6s} {"secs":>5s}  ' + '  '.join(
        f'L={L:<4d}(all/last)' for L in eval_lens)
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
            ev, lab = make_batch(mul, a.batch, a.train_len, a.noise, 1000 + step, dev)
            logits = model(ev)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
        res = eval_lengths(model, mul, eval_lens, a.noise, dev)
        cols = '  '.join(f'{res[L][0]:.3f}/{res[L][1]:.3f}' for L in eval_lens)
        print(f'{arm_spec:16s} {float(loss.detach()):6.3f} {time.time() - t0:5.0f}  {cols}',
              flush=True)
        # If hybrid full-attention, immediately ablate out=0 to test load-bearing!
        if arm_spec == 'full_fullattn':
            saved = {n: mod.out.weight.detach().clone() for n, mod in model.state.items()}
            with torch.no_grad():
                for mod in model.state.values():
                    mod.out.weight.zero_()
            res0 = eval_lengths(model, mul, eval_lens, a.noise, dev)
            cols0 = '  '.join(f'{res0[L][0]:.3f}/{res0[L][1]:.3f}' for L in eval_lens)
            print(f'{"  (out=0 ablate)":16s} {"---":>6s} {"---":>5s}  {cols0}', flush=True)
            with torch.no_grad():
                for n, mod in model.state.items():
                    mod.out.weight.copy_(saved[n])


if __name__ == '__main__':
    main()
