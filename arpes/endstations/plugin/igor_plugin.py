"""Implements a simple loader for Igor files.

This does not load data according to the PyARPES data model, so you should
ideally use a specific data loader where it is available.
"""
from typing import ClassVar

import xarray as xr

from arpes.endstations import (
    SingleFileEndstation,
)
from arpes.load_pxt import read_single_pxt

__all__ = ("IgorEndstation",)


class IgorEndstation(SingleFileEndstation):
    """A generic file loader for PXT files.

    This makes no assumptions about whether data is from a hemisphere
    or otherwise, so it might not be perfect for all Igor users, but it
    is a place to start and to demonstrate how to implement a data loading
    plugin.
    """

    PRINCIPAL_NAME = "Igor"
    ALIASES: ClassVar[list[str]] = [
        "IGOR",
        "pxt",
        "pxp",
        "Wave",
        "wave",
    ]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".pxt",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )

    RENAME_KEYS: ClassVar[dict[str, str | float]] = {}

    MERGE_ATTRS: ClassVar[dict[str, str | float]] = {}

    ATTR_TRANSFORMS: ClassVar[dict[str, str | float]] = {}

    def load_single_frame(
        self,
        frame_path: str | None = None,
        scan_desc: dict | None = None,
        **kwargs,
    ):
        """Igor .pxt and .ibws are single files so we just read the one passed here."""
        print(frame_path, scan_desc)

        pxt_data = read_single_pxt(frame_path)
        return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)
