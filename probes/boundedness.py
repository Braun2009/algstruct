"""PROBE. Boundedness of the composition, without training.

Claim under test (`docs/MODEL_MAP.md` section 2): ||A||_2 <= 1 for every element
and therefore ||Z_T||_2 <= 1 for any horizon T. If this fails, the architecture's
premise fails and nothing downstream matters.

Imports only `model`; never `train`.
"""
from __future__ import annotations

import argparse
import json

import torch

from model import AlgebraConfig, AlgebraState


def run(horizon: int, batch: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = AlgebraConfig(width=192, k=k, state_layers=(0,))
    mod = AlgebraState(cfg)
    a = torch.randn(batch, k, k)
    z = torch.eye(k).expand(batch, -1, -1).clone()
    rec = {'mat': [], 'z': []}
    for _ in range(horizon):
        a = a + .05 * torch.randn_like(a)          # arbitrary walk in gl(k)
        r, p = mod.decompose(a)
        mat = r @ p
        z = mat @ z
        rec['mat'].append(float(torch.linalg.matrix_norm(mat, ord=2).max()))
        rec['z'].append(float(torch.linalg.matrix_norm(z, ord=2).max()))
    return {'max_mat_norm': max(rec['mat']), 'max_z_norm': max(rec['z']),
            'final_z_norm': rec['z'][-1], 'horizon': horizon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizons', default='10,100,1000,3000')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    out = {}
    for h in (int(x) for x in a.horizons.split(',')):
        r = run(h, 32, 3)
        out[h] = r
        print(f'  T={h:<5} max||A||={r["max_mat_norm"]:.6f}  '
              f'max||Z||={r["max_z_norm"]:.6f}  ||Z_T||={r["final_z_norm"]:.4f}')
    if a.out:
        open(a.out, 'w').write(json.dumps(out, indent=2))
    print('bound holds:', all(v['max_mat_norm'] <= 1 + 1e-5 and v['max_z_norm'] <= 1 + 1e-5
                             for v in out.values()))


if __name__ == '__main__':
    main()
