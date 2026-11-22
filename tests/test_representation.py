"""Tests for the algebraic structure stream.

Each test corresponds to one line of the spec:

  representation   non-commutative product; partial inverse (singular => no inverse)
  computation      composition is order sensitive; the stream is a scan, not a mean
  interaction      rho is a homomorphism rho(ab) = rho(a)rho(b)
  no attention     a permuted average of elements is NOT the composition
  generator        the closure quartic is non-negative and vanishes for
                   normal matrices, so it can hold discrete critical points
"""
from __future__ import annotations

import unittest

import torch

from model.action import action_diagnosis, closure_norm, random_element, rho


class TestRepresentation(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_product_is_non_commutative(self):
        a, b = random_element(3, 1., 1), random_element(3, 1., 2)
        self.assertGreater(float((a @ b - b @ a).abs().max()), 1e-3)

    def test_partial_inverse_detected(self):
        s = torch.randn(6)
        good = random_element(3, .5, 3) + 3 * torch.eye(3)
        bad = torch.zeros(3, 3)
        bad[0, 1] = 1.
        ok = action_diagnosis(good, s)
        self.assertTrue(ok['invertible'])
        self.assertLess(ok['inverse_err'], 1e-8)
        broken = action_diagnosis(bad, s)
        self.assertFalse(broken['invertible'])
        self.assertIsNone(broken['inverse_err'])

    def test_closure_quartic_is_non_negative_and_detects_non_normality(self):
        torch.manual_seed(0)
        for _ in range(20):
            a = random_element(3)
            self.assertGreaterEqual(float(closure_norm(a)), -1e-9)
        sym = torch.randn(3, 3)
        self.assertLess(float(closure_norm(.5 * (sym + sym.T))), 1e-12)
        self.assertGreater(float(closure_norm(torch.tensor([[0., 1., 0.],
                                                            [0., 0., 0.],
                                                            [0., 0., 0.]]))), 1e-3)


class TestComputation(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_composition_is_order_sensitive(self):
        a, b = random_element(3, .3, 11), random_element(3, .3, 12)
        s = torch.randn(6)
        ab = rho(a @ b, s)
        ba = rho(b @ a, s)
        self.assertGreater(float((ab - ba).abs().max()), 1e-4)

    def test_averaging_is_not_composition(self):
        # the reason structure must not go through attention: the mean of the
        # products differs from the product of the means
        a, b = random_element(3, .3, 21), random_element(3, .3, 22)
        s = torch.randn(6)
        true = rho(a @ b, s)
        naive = rho(.5 * (a + b), s)
        self.assertGreater(float((true - naive).abs().max()), 1e-3)

    def test_homomorphism(self):
        s = torch.randn(6)
        a, b = random_element(3, .4, 41), random_element(3, .4, 42)
        d = action_diagnosis(a, s)
        self.assertLess(d['homomorphism_err'], 1e-10)
        self.assertGreater(d['noncommutativity'], 1e-3)


if __name__ == '__main__':
    unittest.main()
