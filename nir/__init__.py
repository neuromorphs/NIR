"""The Neuromorphic Intermediate Representation reference implementation.

Documentation: https://nnir.readthedocs.io
"""

from importlib.metadata import version as metadata_version, PackageNotFoundError

try:
    __version__ = version = metadata_version("nir")
    del metadata_version
except PackageNotFoundError:
    # package is not installed
    pass

from . import ir
from .ir import *  # noqa: F403
from .ir import typing  # noqa: F401
from .serialization import read, write

__all__ = ir.__all__ + ["read", "write"]
