"""The architecture. Nothing here imports training code or probes.

    AlgebraConfig   shapes
    AlgebraState    encoding (a in gl(k)), computation (composition of A),
                    coupling (drive into the law, readout out of Z)
    AlgebraModel    backbone + structural stream
"""
from .algebra import ARMS, AlgebraConfig, AlgebraModel, AlgebraState, cayley
from .action import action_diagnosis, closure_norm, rho, random_element
from .block import LocalBlock

__all__ = ['ARMS', 'AlgebraConfig', 'AlgebraModel', 'AlgebraState', 'cayley',
           'LocalBlock', 'rho', 'action_diagnosis', 'closure_norm', 'random_element']
