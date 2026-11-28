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
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import ARMS, AlgebraConfig, AlgebraModel
from .bench import TASKS, VOCAB, dataset


def batch(samples, seq, diff, seed, device, tasks=TASKS):
    ev, lab, tid = dataset(samples, seq, diff, seed, tasks=tasks)[:3]
    return ev.long().to(device), lab.long().to(device), tid.to(device)


@torch.no_grad()
def evaluate(model, seq, diff, seed, device, tasks=TASKS, n=320):
    """Accuracy, plus the two numbers that make accuracy interpretable.

    The bench builds examples in pairs: same program, one decisive event
    differs, the two labels differ. So a model that reads everything except
    that event gives the *same* answer for both members, is right on exactly
    one of them, and scores exactly 0.5 no matter how well it understands the
    rest. Measured on register_swap: predictions identical for both members in
    100% of pairs, accuracy 0.539/0.457 -> mean 0.498, and 0.996 of pairs had
    exactly one member right.

    An accuracy of 0.5 is therefore not "half learned": it is the signature of
    a model that ignores the decisive event completely. Two numbers separate
    the cases:

      flip   fraction of pairs where the two members get different answers.
             0 means the decisive event is invisible to the model and the
             accuracy carries no information at all.
      dir    among flipped pairs, the fraction where the right member was
             picked. 0.5 means it noticed the difference but guesses the
             direction, which is still chance.

    Only when flip is high and dir is above 0.5 is `acc` a measurement of
    anything.
    """
    model.eval()
    ev, lab, tid, ev2, lab2 = dataset(n, seq, diff, seed, tasks=tasks, paired=True)
    ev, lab, tid = ev.long().to(device), lab.long().to(device), tid.to(device)
    ev2, lab2 = ev2.long().to(device), lab2.long().to(device)
    stats = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        p1 = model(ev, stats)[:, -1].argmax(-1)
        p2 = model(ev2)[:, -1].argmax(-1)
    r1, r2 = (p1 == lab), (p2 == lab2)
    flip = (p1 != p2)

    acc, flips, dirs = {}, {}, {}
    for i, name in enumerate(tasks):
        m_i = tid == i
        if not bool(m_i.any()):
            continue
        # mean accuracy over the two members: the number the old table reported
        acc[name] = float((r1[m_i].float().mean() + r2[m_i].float().mean()) / 2)
        flips[name] = float(flip[m_i].float().mean())
        if bool(flip[m_i].any()):
            dirs[name] = float(r1[m_i][flip[m_i]].float().mean())
        else:
            dirs[name] = float('nan')
    inv = stats
    model.train()
    return acc, inv, flips, dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='full', choices=ARMS)
    ap.add_argument('--width', type=int, default=192)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--window', type=int, default=0,
                    help='0 means full causal attention; the bench scatters events over the whole sequence')
    ap.add_argument('--state-layers', default='1,3')
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--sequence', type=int, default=64)
    ap.add_argument('--difficulty', type=int, default=3)
    ap.add_argument('--tasks', default='',
                    help='comma list of bench tasks; empty means all fifteen. '
                         'Single-task runs are how a mixture effect is separated '
                         'from "this task is not learnable at this budget".')
    ap.add_argument('--eval-difficulty', default='',
                    help='comma list; defaults to the training difficulty. '
                         'values above it are a harder held-out setting')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--every', type=int, default=200)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = torch.device(a.device)
    tasks = tuple(x.strip() for x in a.tasks.split(',') if x.strip()) or TASKS
    torch.manual_seed(a.seed)

    cfg = AlgebraConfig(width=a.width, layers=a.layers, k=a.k, window=a.window or None,
                        state_layers=tuple(int(x) for x in a.state_layers.split(',')),
                        arm=a.arm)
    model = AlgebraModel(cfg, VOCAB).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=.01)
    # 3e-3 on a 460M model pins loss at 3.7 and accuracy at chance; the same
    # rate is fine on 7M. Warmup then cosine.
    def lr_lambda(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        # must return a float: a tensor multiplier silently drove the LR to 0
        return .5 * (1 + math.cos(math.pi * min(1., prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(json.dumps({'event': 'start', 'args': {**vars(a), 'tasks': list(tasks)},
                      'params': sum(p.numel() for p in params)}), flush=True)

    t0, hist = time.time(), []
    for step in range(1, a.steps + 1):
        ev, lab, _ = batch(a.batch, a.sequence, a.difficulty,
                           10_000 + step + 7919 * a.seed, dev, tasks)
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
            diffs = ([int(x) for x in a.eval_difficulty.split(',')]
                     if a.eval_difficulty else [a.difficulty])
            results = {}
            for d in diffs:
                results[d] = evaluate(model, a.sequence, d, 555 + a.seed, dev, tasks)
            acc, inv, flips, dirs = results[diffs[0]]
            row = {'event': 'progress', 'step': step, 'loss': float(loss.detach()),
                   'mean_acc': sum(acc.values()) / len(acc),
                   'mean_flip': sum(flips.values()) / len(flips),
                   'seconds': round(time.time() - t0, 1),
                   'inv': {k: round(v, 3) for k, v in inv.items()},
                   'dead_params': sum(1 for p in params if p.grad is None) if step == a.steps else None}
            row['acc'] = {k: round(v, 3) for k, v in acc.items()}
            # the two numbers that decide whether acc means anything
            row['flip'] = {k: round(v, 3) for k, v in flips.items()}
            row['dir'] = {k: (None if v != v else round(v, 3)) for k, v in dirs.items()}
            if len(diffs) > 1:
                row['acc_by_difficulty'] = {
                    d: round(sum(results[d][0].values()) / len(results[d][0]), 4)
                    for d in diffs}
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
