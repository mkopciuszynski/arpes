"""Top level module for PyARPES."""

# pylint: disable=unused-import
from __future__ import annotations

from .check import check

# Use both version conventions for people's sanity.
VERSION = "4.0.0 beta1"
__version__ = VERSION


__all__ = ["__version__"]
