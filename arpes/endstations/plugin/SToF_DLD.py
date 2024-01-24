"""Provides data loading for the Lanzara group experimental ARToF."""
from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import xarray as xr

import arpes.config
from arpes.endstations import SCANDESC, EndstationBase
from arpes.provenance import provenance_from_file

if TYPE_CHECKING:
    from _typeshed import Incomplete

__all__ = ("SToFDLDEndstation",)


class SToFDLDEndstation(EndstationBase):
    """Provides data loading for the Lanzara group experimental ARToF."""

    PRINCIPAL_NAME = "ALG-SToF-DLD"

    def load(self, scan_desc: SCANDESC | None = None, **kwargs: Incomplete) -> xr.Dataset:
        """Load a FITS file containing run data from Ping and Anton's delay line detector ARToF.

        Params:
            scan_desc: Dictionary with extra information to attach to the xarray.Dataset,
            must contain the location of the file

        Returns:
            The loaded spectrum.
        """
        if scan_desc is None:
            warnings.warn(
                "Attempting to make due without user associated metadata for the file",
                stacklevel=2,
            )
            msg = "Expected a dictionary of metadata with the location of the file"
            raise TypeError(msg)
        if kwargs:
            warnings.warn("Any kwargs is not supported.", stacklevel=2)

        metadata = copy.deepcopy(scan_desc)

        data_loc = Path(metadata["file"])
        if not data_loc.is_absolute():
            assert arpes.config.DATA_PATH is not None
            data_loc = Path(arpes.config.DATA_PATH) / data_loc

        f = h5py.File(data_loc, "r")

        dataset_contents = {}
        raw_data = f["/PRIMARY/DATA"][:]
        raw_data = raw_data[:, ::-1]  # Reverse the timing axis
        dataset_contents["raw"] = xr.DataArray(
            raw_data,
            coords={"x_pixels": np.linspace(0, 511, 512), "t_pixels": np.linspace(0, 511, 512)},
            dims=("x_pixels", "t_pixels"),
            attrs=f["/PRIMARY"].attrs.items(),
        )

        provenance_from_file(
            dataset_contents["raw"],
            str(data_loc),
            {
                "what": "Loaded Anton and Ping DLD dataset from HDF5.",
                "by": "load_DLD",
            },
        )

        return xr.Dataset(dataset_contents, attrs=metadata)
