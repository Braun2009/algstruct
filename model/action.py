"""The action rho: how the structural algebra acts on the semantic vector.

    rho(a) s = a @ s   (s viewed as k x q)

Left multiplication is the canonical module action: it is a homomorphism,
rho(ab) = rho(a) rho(b), it is non-commutative, and its inverse exists exactly
when a is invertible -- a *partial* inverse, which is the point.

E/B is one special case of an action, never an assumption.
"""
from __future__ import annotations

import torch


def closure_norm(a):
    """||[a, a^T]||_F^2 >= 0, zero iff a is normal. Non-negative for any a.

    This is the Yang-Mills-type quartic and the natural invariant to put in a
    potential on an algebra: it penalises non-normality, so its critical points
    are the normal matrices, which is a much richer set than the symmetric ones.
    """
    c = a @ a.transpose(-1, -2) - a.transpose(-1, -2) @ a
    return (c * c).sum((-1, -2))


def rho(a, s, arm='linear'):
    """a (...,k,k), s (...,m) with m = k*q  ->  (...,m). Left multiplication.

    No parameters: this is the canonical module action. It is a homomorphism,
    rho(ab) = rho(a) rho(b), which is exactly what makes it a *coupling* rather
    than an additive force.
    """
    k = a.shape[-1]
    if s.shape[-1] % k:
        raise ValueError('semantic width must be a multiple of k')
    q = s.shape[-1] // k
    sv = s.reshape(*s.shape[:-1], k, q)
    if arm == 'linear':
        return (a @ sv).reshape(*s.shape)
    if arm == 'commutator':
        return (a @ sv - sv @ a).reshape(*s.shape) if q == k else \
            (a @ sv - sv.mean(-1, keepdim=True) * 0).reshape(*s.shape)
    raise ValueError('invalid action')


def action_diagnosis(a, s, eps=1e-8):
    """Is the action legal? Homomorphism, non-commutativity, partial inverse.

    Returns a dict of numbers, not booleans, so they can be reported.
    """
    k = a.shape[-1]
    b = torch.randn_like(a)
    left = rho(a @ b, s)
    right = rho(a, rho(b, s))
    hom = float((left - right).abs().max() / (right.abs().max() + eps))
    swap = float((rho(a, rho(b, s)) - rho(b, rho(a, s))).abs().max()
                 / (right.abs().max() + eps))
    det = torch.linalg.det(a.double())
    cond = float(torch.linalg.cond(a.double()))
    out = {'homomorphism_err': hom, 'noncommutativity': swap,
           'det': float(det), 'cond': cond, 'invertible': bool(abs(float(det)) > eps)}
    if out['invertible']:
        inv = torch.linalg.inv(a.double())
        back = rho(inv, rho(a, s.double()))
        out['inverse_err'] = float((back - s.double()).abs().max())
    else:
        out['inverse_err'] = None
    return out


def random_element(k, scale=1.0, seed=None):
    """A random element of gl(k); used by the tests and by probes/coverage.py."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return scale * torch.randn(k, k, generator=g)
