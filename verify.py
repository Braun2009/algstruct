"""Structural invariants of the model, checked numerically.

Lives outside `model/` on purpose: the architecture must not import probes, and
a probe must not be able to change the thing it measures.

    python verify.py
    ALG_EAGER=1 python verify.py

Every check is a number, not a boolean where a number is more informative.
"""
from __future__ import annotations

import math
import sys

import torch

from model import AlgebraConfig, AlgebraModel, RoPE, cayley, rho, rho_split
from model.algebra import Generator, _relax_step


def hdr(name):
    print(f'\n== {name}')


def check(label, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {label}{"  " + detail if detail else ""}')
    return ok


def action_algebra():
    hdr('the action is a representation')
    torch.manual_seed(0)
    k, q, n = 8, 16, 5
    a, b = torch.randn(n, k, k), torch.randn(n, k, k)
    s = torch.randn(n, k, q)
    hom = (rho_split(a @ b, s) - rho_split(a, rho_split(b, s))).abs().max()
    nc = (rho_split(a, rho_split(b, s)) - rho_split(b, rho_split(a, s))).abs().max()
    mean = (rho_split(.5 * (a + b), s) - .5 * (rho_split(a, s) + rho_split(b, s))).abs().max()
    ok = check('rho(ab) = rho(a) rho(b)', float(hom) < 1e-5, f'max err {float(hom):.2e}')
    ok &= check('non-commutative', float(nc) > 1e-3, f'swap diff {float(nc):.2e}')
    ok &= check('linear in the matrix', float(mean) < 1e-5, f'max err {float(mean):.2e}')
    flat = torch.randn(n, k * q)
    same = (rho(a, flat) - rho_split(a, flat.reshape(n, k, q)).reshape(n, k * q)).abs().max()
    ok &= check('flat == split', float(same) < 1e-6, f'max err {float(same):.2e}')
    return ok


def transform_bounds():
    hdr('the per-token transform is a contraction')
    torch.manual_seed(1)
    cfg = AlgebraConfig(width=192, layers=2, k=6, q=32, state_layers=(0,))
    m = AlgebraModel(cfg, 64)
    ev = torch.randint(0, 64, (3, 64, 4))
    layer = m.state['0']
    cap = {}
    orig = layer.decompose

    def spy(a, tau):
        r, p = orig(a, tau)
        cap['r'], cap['p'] = r, p
        return r, p
    layer.decompose = spy
    with torch.no_grad():
        m(ev)
    layer.decompose = orig
    r, p = cap['r'], cap['p']
    rn = torch.linalg.matrix_norm(r, ord=2).max()
    pn = torch.linalg.matrix_norm(p, ord=2).max()
    orth = (r.mT @ r - torch.eye(6)).abs().max()
    det = torch.linalg.det(r.double())
    ok = check('||R||_2 <= 1', float(rn) <= 1 + 1e-5, f'max {float(rn):.6f}')
    ok &= check('||P||_2 <= 1', float(pn) <= 1 + 1e-5, f'max {float(pn):.6f}')
    ok &= check('R orthogonal', float(orth) < 1e-5, f'max dev {float(orth):.2e}')
    ok &= check('det R = +1 (SO(k), not O(k))', float((det - 1).abs().max()) < 1e-5,
                f'max |det-1| {float((det - 1).abs().max()):.2e}')
    return ok


def exact_no_op():
    hdr('a token with tau = 0 is an exact no-op')
    # tau = dt * sigmoid(bias); bias = -60 gives tau ~ 1e-26, so R = P = I
    torch.manual_seed(2)
    m = AlgebraModel(AlgebraConfig(width=192, layers=2, k=6, q=32,
                                   state_layers=(0,), tau_bias=-60.0), 64)
    st = {}
    ev = torch.randint(0, 64, (2, 48, 4))
    with torch.no_grad():
        m(ev, st)
    ok = check('tau -> 0', st['tau_mean'] < 1e-20, f'tau {st["tau_mean"]:.2e}')
    ok &= check('Z_t -> I', abs(st['z_ratio'] - 1.0) < 1e-6,
                f'z_ratio {st["z_ratio"]:.8f} (1.0 = identity product)')
    # a_t stays nonzero on purpose: it is the token's local generator, and tau is
    # the *amount* of it that is applied. A_t = exp(tau * ...) = I either way.
    ok &= check('a_t is not forced to 0 (tau scales the transform, not a)',
                st['a_scale'] > 1e-3, f'|a| {st["a_scale"]:.3f}')
    return ok


def generator_coercivity():
    hdr('the generator is coercive and stays traceless')
    g = Generator()
    v, d0, d1, w0, w1 = g.coeffs()
    ok = check('v3 > v2', float(v[1]) > float(v[0]),
               f'v2={float(v[0]):.4f} v3={float(v[1]):.4f}')
    # purely antisymmetric a: V = (v3 - v2)|a|^2 and grad V = 2(v3 - v2) a
    for scale in (0.1, 0.5, 1.0, 3.0):
        a = torch.zeros(1, 6, 6)
        a[0, :3, 3:] = torch.eye(3) * scale
        a[0, 3:, :3] = -torch.eye(3) * scale
        c = a @ a.mT - a.mT @ a
        V = (v[0] * (a @ a).diagonal(dim1=-2, dim2=-1).sum(-1)
             + v[1] * a.square().sum((-1, -2)) + v[2] * c.square().sum((-1, -2)))
        grad = 2 * v[0] * a.mT + 2 * v[1] * a + 4 * v[2] * (c @ a - a @ c)
        fro = float(a.norm())
        exp_v = (float(v[1]) - float(v[0])) * fro ** 2
        exp_g = 2 * (float(v[1]) - float(v[0])) * fro
        print(f'    |a|={fro:6.3f}  V={float(V):9.4f} (exp {exp_v:9.4f})  '
              f'|grad|={float(grad.norm()):8.4f} (exp {exp_g:8.4f})')
        ok &= abs(float(V) - exp_v) < 1e-3 * max(1, exp_v)
        ok &= abs(float(grad.norm()) - exp_g) < 1e-3 * max(1, exp_g)
    # traceless after relaxation
    torch.manual_seed(3)
    m = AlgebraModel(AlgebraConfig(width=192, layers=2, k=6, q=32,
                                   state_layers=(0,)), 64)
    layer = m.state['0']
    cap = {}
    orig = layer.decompose

    def spy2(a, tau):
        cap['a'] = a
        return orig(a, tau)
    layer.decompose = spy2
    with torch.no_grad():
        m(torch.randint(0, 64, (2, 64, 4)))
    layer.decompose = orig
    tr = cap['a'].diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()
    ok &= check('tr(a) = 0', float(tr) < 1e-6, f'max |tr| {float(tr):.2e}')
    return ok


def stream_bounds():
    hdr('bounds along the sequence')
    torch.manual_seed(4)
    k = 6
    allok = True
    for T in (64, 256, 1024):
        m = AlgebraModel(AlgebraConfig(width=192, layers=2, k=k, q=32,
                                       state_layers=(0,), arm='full'), 64)
        st = {}
        with torch.no_grad():
            m(torch.randint(0, 64, (2, T, 4)), st)
        ok = st['z_ratio'] <= 1 + 1e-5
        ok &= abs(st['R_norm_last'] - math.sqrt(k)) < 1e-3
        allok &= ok
        print(f'  T={T:<5} z_ratio={st["z_ratio"]:.4f}  '
              f'R_norm={st["R_norm_last"]:.4f} (sqrt k = {math.sqrt(k):.4f})  '
              f'a={st["a_scale"]:.3f}  tau={st["tau_mean"]:.3f}  '
              f'{"ok" if ok else "FAIL"}')
    return allok


def no_cross_token_recurrence():
    hdr('encoding is per-token: no cross-token recurrence')
    # Changing token t must not change a_s for s < t. Checked by comparing the
    # a_t trajectory of two batches that agree up to position 5.
    torch.manual_seed(5)
    m = AlgebraModel(AlgebraConfig(width=192, layers=2, k=6, q=32,
                                   state_layers=(0,)), 64)
    layer = m.state['0']
    cap = {}

    def grab(ev):
        orig = layer.decompose

        def spy(a, tau):
            cap['a'] = a.detach()
            return orig(a, tau)
        layer.decompose = spy
        with torch.no_grad():
            m(ev)
        layer.decompose = orig
        return cap['a']

    ev = torch.randint(0, 64, (2, 16, 4))
    a1 = grab(ev)
    ev2 = ev.clone()
    ev2[:, 10:] = torch.randint(0, 64, ev2[:, 10:].shape)
    a2 = grab(ev2)
    early_same = torch.allclose(a1[:, :10], a2[:, :10], atol=0)
    late_diff = not torch.allclose(a1[:, 10:], a2[:, 10:], atol=1e-6)
    ok = check('a_0..a_9 unchanged when tokens 10..15 change', bool(early_same))
    ok &= check('a_10..a_15 do change', bool(late_diff))
    return ok


def init_behaviour():
    hdr('what the structural path does at initialisation')
    torch.manual_seed(6)
    cfg = dict(width=192, layers=4, k=6, q=32, state_layers=(1, 3))
    mf = AlgebraModel(AlgebraConfig(arm='full', out_init=0.0, **cfg), 64)
    mo = AlgebraModel(AlgebraConfig(arm='off', **cfg), 64)
    shared = {k: v for k, v in mo.state_dict().items() if k in mf.state_dict()}
    mf.load_state_dict(shared, strict=False)
    ev = torch.randint(0, 64, (2, 32, 4))
    with torch.no_grad():
        d0 = (mf(ev) - mo(ev)).abs().max()
    ok = check('out_init = 0 -> full == off exactly', float(d0) < 1e-6,
               f'max |d| {float(d0):.2e}')
    torch.manual_seed(6)
    mf2 = AlgebraModel(AlgebraConfig(arm='full', out_init=3e-3, **cfg), 64)
    mf2.load_state_dict(shared, strict=False)
    with torch.no_grad():
        d1 = (mf2(ev) - mo(ev)).abs().max()
    ok &= check('out_init > 0 -> path is live at step 0', float(d1) > 1e-6,
                f'max |d| {float(d1):.2e}')
    return ok


def rope_properties():
    hdr('rotary positions')
    rope = RoPE(64)
    x = torch.randn(1, 1, 8, 64)
    q = rope(x)
    ok = check('preserves norm', bool(torch.allclose(q.norm(dim=-1), x.norm(dim=-1), atol=1e-5)))
    ok &= check('position 0 is the identity', bool(torch.allclose(q[:, :, 0], x[:, :, 0], atol=1e-6)))
    torch.manual_seed(7)
    xi = torch.randn(1, 1, 1, 64)
    yj = torch.randn(1, 1, 1, 64)

    def dot(i, j):
        return float((rope(xi, offset=i) * rope(yj, offset=j)).sum())

    base = dot(0, 3)
    for i, j in ((0, 3), (5, 8), (40, 43)):
        d = abs(dot(i, j) - base)
        ok &= check(f'<q_{i}, k_{j}> == <q_0, k_3> (j-i = 3)', d < 1e-4, f'|d| {d:.2e}')
    gap = abs(dot(0, 3) - dot(0, 9))
    ok &= check('depends on j - i', gap > 1e-2, f'diff {gap:.3f}')
    return ok


def arm_gradients():
    hdr('every arm gets gradients where it should')
    torch.manual_seed(8)
    from torch.nn import functional as F
    ok = True
    for arm in ('full', 'nocompose', 'nodrive'):
        m = AlgebraModel(AlgebraConfig(width=192, layers=4, k=6, q=32,
                                       state_layers=(1,), arm=arm), 64)
        with torch.no_grad():
            m.state['1'].out.weight.normal_(0, .02)
        ev = torch.randint(0, 64, (8, 64, 4))
        lab = torch.randint(0, 6, (8,))
        loss = F.cross_entropy(m(ev)[:, -1], lab)
        loss.backward()
        gen = [(n, p.grad) for n, p in m.named_parameters() if 'gen' in n]
        finite = all(g is not None and torch.isfinite(g).all() for _, g in gen)
        nonzero = all(g is not None and g.norm() > 0 for _, g in gen)
        dead = [n for n, p in m.named_parameters()
                if p.grad is None or p.grad.abs().sum() == 0]
        # nodrive feeds zeros, so a stays 0, every A_t = I and Z is constant:
        # dn (2), to_drive (2), to_tau (2) and the three law coefficients get no
        # gradient. to_s still does, because the action Z @ s is nonzero even
        # when Z = I. That is the arm's definition, not a defect -- but it must
        # be zero, never NaN.
        expected = 9 if arm == 'nodrive' else 0
        arm_ok = finite and (not nonzero if arm == 'nodrive' else nonzero) \
            and len(dead) == expected
        ok &= arm_ok
        print(f'  [{"ok" if arm_ok else "FAIL"}] {arm:10s} law grads '
              f'{[f"{float(g.norm()):.2e}" if g is not None else "None" for _, g in gen]}  '
              f'dead {len(dead)}' + (' (expected: drive is off)' if arm == 'nodrive' else ''))
        if dead and arm != 'nodrive':
            print(f'        {dead}')
    return ok


def main():
    results = [
        action_algebra(),
        transform_bounds(),
        exact_no_op(),
        generator_coercivity(),
        stream_bounds(),
        no_cross_token_recurrence(),
        init_behaviour(),
        rope_properties(),
        arm_gradients(),
    ]
    print(f'\n{sum(1 for r in results if r)}/{len(results)} groups ok')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
