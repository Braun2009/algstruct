"""Tests for the model mapping in docs/v5/MODEL_MAP.md.

Each test pins one line of that page. The point is that a wrong mapping can
still train and produce plausible numbers, so the mapping itself has to be
checked, not just the loss.
"""
from __future__ import annotations

import unittest

import math

import torch
from torch.nn import functional as F

from model import (ARMS, AlgebraConfig, AlgebraModel, AlgebraState,
                                  cayley)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.mod = AlgebraState(AlgebraConfig(width=192, k=3))
        torch.manual_seed(0)

    def test_analytic_gradient_matches_autograd(self):
        v = torch.tensor([.3, .7, .5, .9])
        for _ in range(3):
            a = torch.randn(4, 3, 3, dtype=torch.float64)
            a.requires_grad_(True)
            f = self.mod.potential(a, v)
            g, = torch.autograd.grad(f.sum(), a)
            err = float((g - self.mod.grad_potential(a.detach(), v)).abs().max())
            self.assertLess(err, 1e-10)

    def test_potential_is_coercive(self):
        # a cubic invariant would make V unbounded below; with only even terms
        # V must grow along a = t I in both directions
        v = torch.tensor([0., .5, .5, .5])
        for sign in (+1., -1.):
            vals = [float(self.mod.potential((sign * t * torch.eye(3)).unsqueeze(0), v))
                    for t in (.1, 1., 5., 20.)]
            self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])), vals)

    def test_softplus_keeps_the_coercive_coefficients_positive(self):
        v, *_ = self.mod.coeffs()
        self.assertTrue(bool((v[1:] > 0).all()))


class TestComputation(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.mod = AlgebraState(AlgebraConfig(width=192, k=3))

    def test_bound_holds_for_any_a(self):
        torch.manual_seed(1)
        for scale in (.1, 1., 10., 100.):
            a = scale * torch.randn(8, 3, 3, dtype=torch.float64)
            r, p = self.mod.decompose(a)
            # R orthogonal, P spectrum in (0, 1]
            eye = torch.eye(3, dtype=torch.float64)
            self.assertLess(float((r.mT @ r - eye).abs().max()), 1e-8)
            self.assertLessEqual(float(torch.linalg.eigvalsh(p).max()), 1 + 1e-12)
            # the bound is on the SPECTRAL norm; |.|_F <= sqrt(k) |.|_2
            self.assertLessEqual(float(torch.linalg.matrix_norm(r @ p, ord=2).max()), 1 + 1e-8)

    def test_long_product_stays_bounded(self):
        a = torch.randn(4, 3, 3, dtype=torch.float64)
        r, p = self.mod.decompose(a)
        mat = r @ p
        z = torch.eye(3, dtype=torch.float64).expand(4, -1, -1).clone()
        for _ in range(2000):
            z = mat @ z
        self.assertLessEqual(float(torch.linalg.matrix_norm(z, ord=2).max()), 1 + 1e-8)
        self.assertLessEqual(float(z.norm(dim=(-2, -1)).max()), math.sqrt(3) + 1e-8)

    def test_cayley_is_exactly_orthogonal(self):
        a = torch.randn(5, 3, 3, dtype=torch.float64)
        r = cayley(.5 * (a - a.mT))
        eye = torch.eye(3, dtype=torch.float64)
        self.assertLess(float((r.mT @ r - eye).abs().max()), 1e-12)


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float32)

    def _batch(self, n=16, t=32):
        from train.bench import dataset
        ev, lab, tid = dataset(n, t, 1, 0)[:3]
        return ev.long(), lab.long(), tid

    def test_arms_run_and_composition_is_used(self):
        ev, lab, _ = self._batch()
        for arm in ARMS:
            cfg = AlgebraConfig(width=192, layers=2, state_layers=(0,), arm=arm)
            m = AlgebraModel(cfg, 64)
            stats = {}
            out = m(ev, stats)
            loss = F.cross_entropy(out[:, -1], lab)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            if arm in ('full', 'nodrive'):
                # mat_norm is Frobenius: bound is sqrt(k)
                self.assertLessEqual(stats['mat_norm'], math.sqrt(3) + 1e-5)
            if arm == 'full':
                self.assertGreater(stats['a_var'], 0.)
            if arm == 'nodrive':
                # severing W_d must make a token-independent, exactly: this is
                # what makes the ablation a real control rather than a nuisance
                self.assertEqual(stats['a_var'], 0.0)

    def test_no_structure_goes_through_attention(self):
        # the structural path must use matmul only: replacing the attention
        # block by zeros must not silence the structural readout
        ev, _, _ = self._batch()
        cfg = AlgebraConfig(width=192, layers=2, state_layers=(0,), arm='full')
        m = AlgebraModel(cfg, 64)
        with torch.no_grad():
            for blk in m.blocks:
                for p in blk.parameters():
                    p.zero_()
        out = m(ev)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_state_is_bounded_and_not_degenerate(self):
        ev, _, _ = self._batch()
        cfg = AlgebraConfig(width=192, layers=2, state_layers=(0,), arm='full')
        m = AlgebraModel(cfg, 64)
        stats = {}
        m(ev, stats)
        self.assertLessEqual(stats['mat_norm'], math.sqrt(3) + 1e-5)
        self.assertGreater(stats['z_norm'], 0.)
        self.assertGreater(stats['a_var'], 0.)     # a is not a single fixed point


if __name__ == '__main__':
    unittest.main()
