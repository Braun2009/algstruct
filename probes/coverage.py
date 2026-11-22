"""PROBE. How many kinds of local dynamics can ONE fixed generator produce?

The question this answers is the user's: "can it generate all coupling modes?"
`theta` is global and fixed, but D(a) and Omega(a) are *fields*, so different
regions of gl(k) can be dissipative, injecting or gyroscopic at once. Coverage
counts which local types appear while a sweeps the algebra:

    sign of D(a)                           dissipative / injecting / critical
    nonzero Omega(a)                       gyroscopic or not
    Jacobian spectrum at the fixed point    node / focus / saddle / centre
    non-normality ||[a, a^T]||             partial inverse available or not

Imports only `model`; never `train`.
"""
from __future__ import annotations

import argparse
import json
import itertools
from collections import Counter

import torch

from model import AlgebraConfig, AlgebraState


def local_type(mod, a, v, d0, d1, w0, w1):
    r2 = a.square().sum()
    dissip = float(d0 + d1 * r2)
    gyro = float(w0 + w1 * r2)
    jac = torch.autograd.functional.jacobian(
        lambda x: mod.flow(x.reshape(3, 3), torch.zeros(3, 3)).reshape(-1),
        a.reshape(-1)).to(torch.float64)
    ev = torch.linalg.eigvals(jac)
    real = ev.real
    imag = ev.imag.abs().max()
    if (real > 1e-6).any() and (real < -1e-6).any():
        spec = 'saddle'
    elif imag > 1e-4 and real.abs().max() < 1e-4:
        spec = 'centre'
    elif (real < 0).all():
        spec = 'stable'
    else:
        spec = 'unstable'
    return {'D': 'inject' if dissip < -1e-6 else ('dissipate' if dissip > 1e-6 else 'critical'),
            'Omega': 'on' if abs(gyro) > 1e-3 else 'off',
            'spectrum': spec,
            'invertible': bool(abs(float(torch.linalg.det(a))) > 1e-8)}


def sweep(args):
    """Coverage is a property of the FAMILY, so theta must be swept too.

    At initialisation theta has d1 = 0 and omega = 0, i.e. D and Omega are
    constants and every region of gl(k) has the same type (pure dissipation).
    A single fixed theta can only hold several types if D and Omega depend on a.
    """
    torch.set_default_dtype(torch.float64)
    mod = AlgebraState(AlgebraConfig(width=192, k=3, state_layers=(0,)))
    rows = []
    for d0v, d1v, w0v, w1v in [(1., 0., 0., 0.),      # initialisation
                               (-1., 1., 0., 0.),     # van der Pol shape
                               (1., .5, .8, 0.),      # gyroscopic
                               (-.5, 1., .5, .3)]:    # injection + gyroscopic
        with torch.no_grad():
            mod.d.copy_(torch.tensor([d0v, d1v], dtype=torch.float64))
            mod.omega.copy_(torch.tensor([w0v, w1v], dtype=torch.float64))
        v, d0, d1, w0, w1 = mod.coeffs()
        # sweep directions x radii: the radius is what moves ||a||^2 across the
        # sign change of D, so a grid on single entries misses the injection zone
        combos = Counter()
        g = torch.Generator().manual_seed(0)
        radii = torch.linspace(.1, args.scale, args.grid + 1)
        for direction in range(6):
            base = torch.randn(3, 3, generator=g, dtype=torch.float64)
            base = base / base.norm()
            for r in radii:
                t = local_type(mod, r * base, v, d0, d1, w0, w1)
                combos[(t['D'], t['Omega'], t['spectrum'])] += 1
        rows.append({'theta': (d0v, d1v, w0v, w1v), 'combos': len(combos),
                     'detail': dict(combos)})
        print(f'  theta d0={d0v:+.1f} d1={d1v:.1f} w0={w0v:+.1f} w1={w1v:.1f}  '
              f'types {len(combos)}  {sorted(combos)}')
    print('max over theta:', max(r['combos'] for r in rows))
    if args.out:
        open(args.out, 'w').write(json.dumps(rows, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=5)
    ap.add_argument('--scale', type=float, default=1.5)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    mod = AlgebraState(AlgebraConfig(width=192, k=3, state_layers=(0,)))
    v, d0, d1, w0, w1 = mod.coeffs()
    vals = torch.linspace(-args.scale, args.scale, args.grid)
    seen, combos = Counter(), Counter()
    total = 0
    for i, j, l in itertools.product(range(3), range(3), range(args.grid)):
        base = torch.zeros(3, 3, dtype=torch.float64)
        base[i, j] += vals[l]
        base[0, 0] += .3
        t = local_type(mod, base, v, d0, d1, w0, w1)
        seen.update([t['D'], t['Omega'], t['spectrum'], 'inv' if t['invertible'] else 'sing'])
        combos[(t['D'], t['Omega'], t['spectrum'])] += 1
        total += 1
    v, d0, d1, w0, w1 = mod.coeffs()
    theta = [*(round(float(x), 3) for x in v), round(float(d0), 3),
             round(float(d1), 3), round(float(w0), 3), round(float(w1), 3)]
    print('theta =', theta)
    print('local types seen:', dict(seen))
    print('distinct (D, Omega, spectrum) combinations:', len(combos), 'of', 3 * 2 * 4, 'possible')
    for k, c in combos.most_common():
        print('  ', k, c)
    print()
    print('== theta sweep ==')
    sweep(args)


if __name__ == '__main__':
    main()
