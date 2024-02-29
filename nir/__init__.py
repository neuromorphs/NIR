"""The Neuromorphic Intermediate Representation reference implementation.

Documentation: https://nnir.readthedocs.io
"""

from .ir import *
from .ir import typing
from .read import read
from .write import write

__all__ = ir.__all__ + [
    "ir",
    "typing",
    "read",
    "write"
]

version = __version__ = "1.0.1"
