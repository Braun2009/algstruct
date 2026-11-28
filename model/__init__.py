"""The architecture. Nothing here imports training code or probes.

    AlgebraConfig   shapes
    Generator       the 7 coefficients of the law, shared by every layer
    AlgebraState    encoding (a in gl(k)), computation (composition of A),
                    action (rho: Z acts on a structural space of its own)
    AlgebraModel    backbone + structural stream
"""
from .algebra import (ARMS, AlgebraConfig, AlgebraModel, AlgebraState,
                      Generator, cayley)
from .action import action_diagnosis, closure_norm, rho, rho_split, random_element
from .block import LocalBlock, RoPE

__all__ = ['ARMS', 'AlgebraConfig', 'AlgebraModel', 'AlgebraState', 'Generator',
           'cayley', 'LocalBlock', 'RoPE', 'rho', 'rho_split',
           'action_diagnosis', 'closure_norm', 'random_element']
