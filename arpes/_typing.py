"""Specialized type annotations for use in PyARPES.

In particular, we frequently allow using the `DataType` annotation,
which refers to either an xarray.DataArray or xarray.Dataset.

Additionally, we often use `NormalizableDataType` which
means essentially anything that can be turned into a dataset,
for instance by loading from the cache using an ID, or which is
literally already data.
"""
from __future__ import annotations

import uuid
from typing import TypeVar, TypedDict, Literal

import xarray as xr

__all__ = [
    "DataType",
    "NormalizableDataType",
    "xr_types",
    "SPECTROMETER",
    "MOMENTUM",
    "EMISSION_ANGLE",
    "ANGLE",
]

DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)
NormalizableDataType = DataType | str | uuid.UUID

xr_types = (xr.DataArray, xr.Dataset)


MOMENTUM = Literal["kp", "kx", "ky", "kz"]
EMISSION_ANGLE = Literal["phi", "psi"]
ANGLE = Literal["alpha", "beta", "chi", "theta"] | EMISSION_ANGLE


class SPECTROMETER(TypedDict, total=False):
    name: str
    rad_per_pixel: float
    type: str
    is_slit_vertical: bool
    dof: list[str]
    scan_dof: list[str]
    mstar: float
    dof_type: dict[str, list[str]]
    length: float