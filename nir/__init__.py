"""The Neuromorphic Intermediate Representation reference implementation.

Documentation: https://nnir.readthedocs.io
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

try:
    __version__ = version = metadata_version("nir")
    del metadata_version
except PackageNotFoundError:
    # package is not installed
    pass

from . import data_ir, ir
from .data_ir import *
from .ir import *
from .ir import typing  # noqa: F401
from .serialization import read, read_data, write, write_data

__all__ = ir.__all__ + data_ir.__all__ + ["read", "write", "read_data", "write_data"]
