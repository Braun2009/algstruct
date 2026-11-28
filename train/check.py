"""Does the architecture work, and can it learn?

The claim this file tests is narrow and checkable:

  1. the pipeline trains at all -- loss falls, accuracy rises above the paired
     floor, and the *flip* rate rises with it
  2. the structural path carries the decisive event -- flip > 0, dir > 0.5
  3. the arms differ where they should and not where they should not

`acc` alone cannot show any of this on a paired bench: a model that ignores the
one event the pair differs in scores exactly 0.5 while understanding everything
else (measured: predictions identical for both members in 100% of pairs,
acc 0.539/0.457 -> 0.498). So every row reports flip and dir next to acc.

    python -m train.check --tasks group_state --diff 2 --steps 1500
    python -m train.check --tasks register_swap --diff 2 --arms off,full
"""
from __future__ import annotations

import argparse
import math
import time

import torch
from torch.nn import functional as F

from model import ARMS, AlgebraConfig, AlgebraModel
from .bench import dataset


def evaluate(model, task, diff, seed, device, n=512):
    ev, lab, _, ev2, lab2 = dataset(n, 128, diff, seed, tasks=(task,), paired=True)
    ev, lab = ev.long().to(device), lab.long().to(device)
    ev2, lab2 = ev2.long().to(device), lab2.long().to(device)
    model.eval()
    with torch.no_grad():
        p1 = model(ev)[:, -1].argmax(-1)
        p2 = model(ev2)[:, -1].argmax(-1)
    model.train()
    r1, r2, flip = p1 == lab, p2 == lab2, p1 != p2
    return {
        'acc': float((r1.float().mean() + r2.float().mean()) / 2),
        'flip': float(flip.float().mean()),
        'dir': float(r1[flip].float().mean()) if bool(flip.any()) else float('nan'),
    }


def train_one(arm, task, diff, steps, batch, seq, lr, seed, device,
              window=None, k=8, q=48, memory=40.0, quiet=False):
    torch.manual_seed(seed)
    model = AlgebraModel(AlgebraConfig(width=768, layers=12, k=k, q=q,
                                       state_layers=(4, 8), arm=arm,
                                       window=window, memory=memory), 64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: .5 * (1 + math.cos(math.pi * min(1., s / steps))))
    t0, hist = time.time(), []
    for step in range(1, steps + 1):
        ev, lab, _ = dataset(batch, seq, diff, 1000 + step, tasks=(task,))[:3]
        ev, lab = ev.long().to(device), lab.long().to(device)
        loss = F.cross_entropy(model(ev)[:, -1], lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        sched.step()
        if not quiet and step % (steps // 3) == 0:
            hist.append((step, round(float(loss.detach()), 3)))
    res = evaluate(model, task, diff, 777 + seed, device)
    res['loss'] = float(loss.detach())
    res['secs'] = time.time() - t0
    res['hist'] = hist
    res['law'] = model.generators()
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='group_state')
    ap.add_argument('--arms', default='off,full')
    ap.add_argument('--diff', type=int, default=2)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--sequence', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--window', type=int, default=0,
                    help='0 = full causal attention. 1 = each position sees only '
                         'itself, so the only long-range channel is the structural '
                         'state. That is the topology in which the stream is '
                         'necessary rather than optional.')
    ap.add_argument('--k', type=int, default=8, help='algebra dimension')
    ap.add_argument('--q', type=int, default=48,
                    help='bandwidth per component; the interface is k*q wide, so '
                         'k*q versus width 768 is the interface bottleneck')
    ap.add_argument('--memory', type=float, default=40.0,
                    help='tokens the state can integrate; v3 = 1/(2 memory dt)')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    arms = [x for x in a.arms.split(',') if x]
    seeds = [int(x) for x in a.seeds.split(',') if x]

    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f'unknown arm {arm}')
    print(f'diff {a.diff}  steps {a.steps}  batch {a.batch}  seq {a.sequence}  '
          f'lr {a.lr}  window {a.window or "full"}  k {a.k}  q {a.q}  '
          f'(interface {a.k * a.q} wide)  memory {a.memory}')
    print(f'{"task":16s} {"arm":10s} {"seed":>4s} {"loss":>7s} {"acc":>6s} '
          f'{"flip":>6s} {"dir":>6s} {"secs":>6s}')
    for task in [x for x in a.tasks.split(',') if x]:
        for arm in arms:
            for seed in seeds:
                _, r = train_one(arm, task, a.diff, a.steps, a.batch,
                                 a.sequence, a.lr, seed, dev,
                                 window=a.window or None, k=a.k, q=a.q,
                                 memory=a.memory)
                d = '   nan' if r['dir'] != r['dir'] else f'{r["dir"]:6.3f}'
                print(f'{task:16s} {arm:10s} {seed:4d} {r["loss"]:7.3f} '
                      f'{r["acc"]:6.3f} {r["flip"]:6.3f} {d} {r["secs"]:6.0f}', flush=True)
                if arm != 'off':
                    g = r['law']
                    print(f'{"":16s} {"":10s}       law v2={g["v2"]:.4f} v3={g["v3"]:.4f} '
                          f'memory={1 / (0.1 * 2 * g["v3"]):.0f} tokens')


if __name__ == '__main__':
    main()
