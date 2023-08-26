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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Required, TypedDict, TypeVar

import xarray as xr

## from Matplotlib 3.8dev
## After 3.8 release, the two lines below should be removed.
##
from matplotlib._enums import CapStyle, JoinStyle
from matplotlib.markers import MarkerStyle

__all__ = [
    "DataType",
    "NormalizableDataType",
    "xr_types",
    "SPECTROMETER",
    "MOMENTUM",
    "EMISSION_ANGLE",
    "ANGLE",
    "NAN_POLICY",
    "CONFIGTYPE",
    "ARPESGeneralAttrs",
]

DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)
NormalizableDataType = DataType | str | uuid.UUID

xr_types = (xr.DataArray, xr.Dataset)


MOMENTUM = Literal["kp", "kx", "ky", "kz"]
EMISSION_ANGLE = Literal["phi", "psi"]
ANGLE = Literal["alpha", "beta", "chi", "theta"] | EMISSION_ANGLE
NAN_POLICY = Literal["raise", "propagate", "omit"]


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
    ##
    analyzer: str
    analyzer_name: str
    parallel_deflectors: bool
    perpendicular_deflectors: bool
    analyzer_radius: int | float
    analyzer_type: str
    mcp_voltage: float | int | None
    probe_linewidth: float
    alpha: float
    chi: float
    theta: float
    psi: float


class CONFIGTYPE(TypedDict, total=False):
    WORKSPACE: Required[dict[str, str | Path]]
    CURRENT_CONTEXT: Required[str | None]
    ENABLE_LOGGING: Required[bool]
    LOGGING_STARTED: Required[bool]
    LOGGING_FILE: Required[str | Path | None]


class ARPESGeneralAttrs(TypedDict, total=False):
    spectrum_type: str  # Literal
    angle_unit: Literal["Radians", "Degrees"]
    hv: float
    x: float
    y: float
    z: float
    #
    beta_offset: float
    chi_offset: float
    theta_offset: float
    #
    sharman: float
    sherman_function: float
    SHERMAN: float


class ARPESPumpAttrs(TypedDict, total=False):
    pump_wavelength: float
    pump_energy: float
    pump_fluence: float
    pump_pulse_energy: float
    pump_spot_size: tuple[float, float] | float
    pump_profile: None  # TODO: check this property.
    pump_linewidth: float
    pump_temporal_width: float
    pump_polarization: tuple[float, float] | str
    repetition_rate: float


class ARPESProbeAttrs(TypedDict, total=False):
    probe_wavelength: float
    probe_energy: float
    probe_fluence: float
    probe_pulse_energy: float
    probe_spot_size: tuple[float, float] | float
    probe_profile: None  # TODO: check this property.
    probe_linewidth: float
    probe_temporal_width: float
    probe_polarization: tuple[float, float] | str


ARPESLaserAttrs = ARPESPumpAttrs | ARPESProbeAttrs


class ARPESAnalyzerAttrs(TypedDict, total=False):
    lens_mode: str  # Literal
    lens_mode_name: str  # Literal
    acquisition_mode: str
    pass_energy: float
    slit_shape: str
    slit_width: float
    slit_number: float | str
    analyzer_type: str
    mcp_voltage: float
    work_function: float


ARPESAttrs = ARPESLaserAttrs | ARPESAnalyzerAttrs | ARPESGeneralAttrs | ARPESLaserAttrs

RGBColorType = tuple[float, float, float] | str
RGBAColorType = (
    str  # str is "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    | tuple[RGBColorType, float]
    # 2 tuple (color, alpha) representations, not infinitely recursive
    # RGBColorType includes the (str, float) tuple, even for RGBA strings
    | tuple[tuple[float, float, float, float] | float]
    # (4-tuple, float) is odd, but accepted as the outer float overriding A of 4-tuple
)

ColorType = RGBColorType | RGBAColorType

RGBColourType = RGBColorType
RGBAColourType = RGBAColorType
ColourType = ColorType

LineStyleType = str | tuple[float, Sequence[float]]
DrawStyleType = Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
MarkEveryType = (
    None | int | tuple[int, int] | slice | list[int] | float | tuple[float, float] | list[bool]
)

MarkerType = str | Path | MarkerStyle
JoinStyleType = JoinStyle | Literal["miter", "round", "bevel"]
CapStyleType = CapStyle | Literal["butt", "projecting", "round"]

FillStyleType = Literal["full", "left", "right", "bottom", "top", "none"]
RcStyleType = str | dict[str, Any] | Path | list[str | Path | dict[str, Any]]
