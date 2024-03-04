"""The Neuromorphic Intermediate Representation reference implementation.

Documentation: https://nnir.readthedocs.io
"""

from . import ir
from .ir import *  # noqa: F403
from .ir import typing  # noqa: F401
from .serialization import read, write

__all__ = ir.__all__ + ["read", "write"]

version = __version__ = "1.0.1"
