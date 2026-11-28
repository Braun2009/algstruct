"""Is the answer present in the structured state, or does the readout lose it?

The window=1 experiment gives a clean split:

    off   flip 0.000   the decisive event never reaches the output
    full  flip 0.775   the state does carry it
    full  dir  0.214   but the answer is decoded with the wrong sign

`dir` below 0.5 means the information arrives and is then mis-mapped. Two very
different faults look the same from the outside:

  (a) the state at the readout position does not contain the answer, so the
      trunk guesses; or
  (b) the state contains it, but the interface + head cannot extract it.

A linear probe on `vec(Z_T)` separates them. It has no access to the trunk, so if
the probe decodes the label well, the information is there and the fault is in
the readout path. If the probe is at chance, the state never encoded it and the
fault is upstream, in the encoding.

The probe is trained on a disjoint seed from the evaluation set, and reports both
the probe's accuracy and the model's own, on the same examples.

    python -m probes.state_probe --steps 4000 --window 1
"""
from __future__ import annotations

import argparse
import math

import torch
from torch.nn import functional as F

from model import AlgebraConfig, AlgebraModel
from train.bench import dataset
from train.check import evaluate


def collect(model, layer, task, diff, n, seed, device, T=128):
    """Run one forward and capture, at the readout position:

        z      vec(Z_T), taken from the layer's own decompose + prefix product
        logits the model's own output, from the same pass

    Captured by wrapping two internal functions rather than reimplementing the
    stream: a reimplementation that drifts from the real one would make the
    probe measure a different model. Nothing here changes what is computed.
    """
    ev, lab, _ = dataset(n, T, diff, seed, tasks=(task,))[:3]
    ev = ev.long().to(device)
    cap = {}
    orig_dec, orig_pre = layer.decompose, layer._prefix_product

    def dec(a):
        r, p = orig_dec(a)
        cap['r'], cap['p'] = r, p
        return r, p

    layer.decompose = dec
    layer._prefix_product = lambda mats: cap.setdefault('pre', orig_pre(mats))
    try:
        with torch.no_grad():
            logits = model(ev)
    finally:
        layer.decompose, layer._prefix_product = orig_dec, orig_pre
    z = cap['p'][:, -1] @ cap['pre'][:, -1]
    with torch.no_grad():
        x = sum(e(ev[..., i]) for i, e in enumerate(model.enc))
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if str(i) == layer_key_str(layer, model):
                break
        h_in = x[:, -1]
    return h_in.detach(), z.detach(), lab.to(device), logits[:, -1].detach()


def layer_key_str(layer, model):
    for k, v in model.state.items():
        if v is layer:
            return k
    raise KeyError('layer not in model')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='group_state')
    ap.add_argument('--diff', type=int, default=2)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--window', type=int, default=1)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--q', type=int, default=48)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)

    torch.manual_seed(0)
    model = AlgebraModel(AlgebraConfig(width=768, layers=12, k=a.k, q=a.q,
                                       state_layers=(4, 8), arm='full',
                                       window=a.window or None), 64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: .5 * (1 + math.cos(math.pi * min(1., s / a.steps))))
    for step in range(1, a.steps + 1):
        ev, lab, _ = dataset(128, 128, a.diff, 1000 + step, tasks=(a.task,))[:3]
        ev, lab = ev.long().to(dev), lab.long().to(dev)
        loss = F.cross_entropy(model(ev)[:, -1], lab)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()
        if step % 1000 == 0:
            print(f'  step {step}  loss {float(loss.detach()):.3f}', flush=True)

    acc, inv, flips, dirs = evaluate(model, 128, a.diff, 777, dev, (a.task,))
    print(f'\nmodel: acc {acc[a.task]:.3f}  flip {flips[a.task]:.3f}  '
          f'dir {dirs[a.task]:.3f}')

    layer = model.state[str(layer_key(model))]
    h_tr, z_tr, lab_tr, _ = collect(model, layer, a.task, a.diff, 1024, 31, dev)
    h_ev, z_ev, lab_ev, logits_ev = collect(model, layer, a.task, a.diff, 512, 777, dev)

    def probe(feats_tr, feats_ev, tag, epochs=300):
        """A single linear map, trained on a disjoint seed. No hidden layers."""
        d = feats_tr.shape[-1]
        w = torch.zeros(d, 56, device=dev, requires_grad=True)
        b = torch.zeros(56, device=dev, requires_grad=True)
        o = torch.optim.Adam([w, b], lr=1e-2)
        for _ in range(epochs):
            loss = F.cross_entropy(feats_tr @ w + b, lab_tr)
            o.zero_grad(); loss.backward(); o.step()
        with torch.no_grad():
            p = (feats_ev @ w + b).argmax(-1)
            acc = float((p == lab_ev).float().mean())
            mode = float(lab_ev.bincount().max()) / lab_ev.numel()
        print(f'  linear probe on {tag:22s} ({d:5d} dims): acc {acc:.3f}   '
              f'(constant-predictor {mode:.3f})')
        return acc

    print('\nis the answer in the state?')
    probe(z_tr.reshape(len(z_tr), -1), z_ev.reshape(len(z_ev), -1), 'vec(Z_T)')
    probe(h_tr, h_ev, 'trunk h at readout')
    with torch.no_grad():
        own = float((logits_ev.argmax(-1) == lab_ev).float().mean())
    print(f'  the model itself                  : acc {own:.3f}')


def layer_key(model):
    return sorted(model.state.keys(), key=int)[-1]


if __name__ == '__main__':
    main()
