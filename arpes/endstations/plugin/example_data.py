"""Provides a convenience data loader for example data.

Providing example data is essential for ensuring approachability,
but in the past we only provided a single ARPES cut. We now provide
a variety but need to be parsimonious about disk space for downloads.
As a result, this custom loader let's us pretend we store the data in
a higher quality format.
"""


from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

import arpes.xarray_extensions  # noqa : PGH004
from arpes.endstations import HemisphericalEndstation, SingleFileEndstation

if TYPE_CHECKING:
    from _typeshed import Incomplete

__all__ = ["ExampleDataEndstation"]


class ExampleDataEndstation(SingleFileEndstation, HemisphericalEndstation):
    """Loads data from exported .nc format saved by xarray. Used for storing example data."""

    PRINCIPAL_NAME = "example_data"

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".nc"}

    def load_single_frame(
        self,
        frame_path: str | None = None,
        scan_desc: dict | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Loads single file examples.

        Additionally, copies coordinate offsets onto the dataset because we have
        preloaded these for convenience on maps.
        """
        data = xr.open_dataarray(frame_path)
        data = data.astype(np.float64)

        # Process coordinates so that there are no non-dimension coordinates
        # which are not a function of some index. This is for simplicity for beginners.
        replacement_coords = {}
        for cname, coord in data.coords.items():
            if len(coord.values.shape) and cname not in data.dims:
                replacement_coords[cname] = coord.mean().item()

        data = data.assign_coords(**replacement_coords)

        # Wrap into a dataset
        dataset = xr.Dataset({"spectrum": data})
        dataset.S.apply_offsets(data.S.offsets)

        return dataset
