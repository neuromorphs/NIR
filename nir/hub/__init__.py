"""NIR Model Hub functionality.

Provides tools for uploading and downloading NIR models to/from a hub.
"""

from .client import upload, download

__all__ = ["upload", "download"]
