"""The Neuromorphic Intermediate Representation reference implementation.

Documentation: https://nnir.readthedocs.io
"""

from importlib.metadata import version, PackageNotFoundError

try:
        __version__ = version("nir")
except PackageNotFoundError:
        # package is not installed
            pass

from . import ir
from .ir import *  # noqa: F403
from .ir import typing  # noqa: F401
from .serialization import read, write

__all__ = ir.__all__ + ["read", "write"]

