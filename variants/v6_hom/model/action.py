"""The action rho: how the structural algebra acts on the semantic vector.

    rho(a) s = a @ s      (s viewed as k x q)

Left multiplication is the canonical module action: it is a homomorphism,
rho(ab) = rho(a) rho(b), it is non-commutative, and its inverse exists exactly
when a is invertible -- a *partial* inverse, which is the point.

It has no parameters. The form of the action is not learned; what is learned is
the interface around it (which subspace acts as the operand, and where the
result goes). E/B is one special case of an action, never an assumption.
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


def rho_split(a, s, arm='linear'):
    """Left multiplication with `s` already split as (..., k, q).

    No parameters: this is the canonical module action, so
    rho(ab) = rho(a) rho(b) holds exactly, which is what makes it a *coupling*
    rather than an additive force.
    """
    if arm == 'linear':
        return a @ s
    if arm == 'commutator':
        if s.shape[-1] != a.shape[-1]:
            raise ValueError('commutator action needs q == k')
        return a @ s - s @ a
    raise ValueError('invalid action')


def rho(a, s, arm='linear', split=False):
    """Left multiplication: a (...,k,k) acts on s.

    With `split=False` (the default) `s` is a flat semantic vector of width
    m = k*q and is viewed as k rows; the result is flat again. With
    `split=True` it is already (..., k, q) -- the internal form, which avoids
    the ambiguity of a flat width when q == 1.
    """
    k = a.shape[-1]
    if split:
        return rho_split(a, s, arm)
    m = s.shape[-1]
    if m % k:
        raise ValueError('semantic width must be a multiple of k')
    sv = s.reshape(*s.shape[:-1], k, m // k)
    return rho_split(a, sv, arm).reshape(*s.shape)


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
    """A random element of gl(k)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return scale * torch.randn(k, k, generator=g)
