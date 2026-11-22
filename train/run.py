"""Train the algebraic-structure model on the multi-task state-tracking bench.

One model, one shared generator; per-task accuracy is read out from the same
run, so the whole task suite is one training job (not one per task).

    python -m train.run --arm full --steps 1600

Reported besides loss:
  per-task accuracy        what the bench is for
  ||A|| max                the bound from the polar split; must stay <= 1
  ||z||                    must be bounded, not growing with t
  dead parameters          must be 0 (an ablation that silently disables a
                           coefficient is a different experiment, say so)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import ARMS, AlgebraConfig, AlgebraModel
from .bench import TASKS, VOCAB, dataset


def batch(samples, seq, diff, seed, device):
    ev, lab, tid = dataset(samples, seq, diff, seed)[:3]
    return ev.long().to(device), lab.long().to(device), tid.to(device)


@torch.no_grad()
def evaluate(model, seq, diff, seed, device, n=320):
    model.eval()
    ev, lab, tid = batch(n, seq, diff, seed, device)
    stats = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        logits = model(ev, stats)[:, -1]
    right = (logits.argmax(-1) == lab).float()
    acc = {name: float(right[tid == i].mean()) for i, name in enumerate(TASKS)
           if bool((tid == i).any())}
    inv = stats
    model.train()
    return acc, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='full', choices=ARMS)
    ap.add_argument('--width', type=int, default=192)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--state-layers', default='1,3')
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--sequence', type=int, default=64)
    ap.add_argument('--difficulty', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--every', type=int, default=200)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)

    cfg = AlgebraConfig(width=a.width, layers=a.layers, k=a.k, window=a.window,
                        state_layers=tuple(int(x) for x in a.state_layers.split(',')),
                        arm=a.arm)
    model = AlgebraModel(cfg, VOCAB).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    print(json.dumps({'event': 'start', 'args': vars(a),
                      'params': sum(p.numel() for p in params)}), flush=True)

    t0, hist = time.time(), []
    for step in range(1, a.steps + 1):
        ev, lab, _ = batch(a.batch, a.sequence, a.difficulty,
                           10_000 + step + 7919 * a.seed, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=dev.type == 'cuda'):
            logits = model(ev)[:, -1]
        loss = F.cross_entropy(logits.float(), lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        sched.step()
        if step % a.every == 0 or step == a.steps:
            acc, inv = evaluate(model, a.sequence, a.difficulty, 555 + a.seed, dev)
            row = {'event': 'progress', 'step': step, 'loss': float(loss.detach()),
                   'mean_acc': sum(acc.values()) / len(acc),
                   'seconds': round(time.time() - t0, 1),
                   'inv': {k: round(v, 3) for k, v in inv.items()},
                   'dead_params': sum(1 for p in params if p.grad is None) if step == a.steps else None}
            row['acc'] = {k: round(v, 3) for k, v in acc.items()}
            hist.append(row)
            print(json.dumps(row), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'args': vars(a), 'history': hist},
                                          indent=2), encoding='utf-8')
    print(json.dumps({'event': 'done', 'seconds': round(time.time() - t0, 1),
                      'mean_acc': hist[-1]['mean_acc']}), flush=True)


if __name__ == '__main__':
    main()
