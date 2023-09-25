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
from typing import TYPE_CHECKING, Literal, Required, TypedDict, TypeVar

import xarray as xr

# from Matplotlib 3.8dev
# After 3.8 release, the two lines below should be removed.
##

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from _typeshed import Incomplete
    from numpy.typing import NDArray

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
    "WORKSPACETYPE",
    "ANALYZERINFO",
]

DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)
NormalizableDataType = DataType | str | uuid.UUID

xr_types = (xr.DataArray, xr.Dataset)


MOMENTUM = Literal["kp", "kx", "ky", "kz"]
EMISSION_ANGLE = Literal["phi", "psi"]
ANGLE = Literal["alpha", "beta", "chi", "theta"] | EMISSION_ANGLE
NAN_POLICY = Literal["raise", "propagate", "omit"]


class ConfigSettings(TypedDict, total=False):
    interactive: dict[str, float | str]
    xarray_repr_mod: bool
    use_tex: bool


class COORDINATES(TypedDict, total=False):
    x: NDArray[np.float_] | float | None
    y: NDArray[np.float_] | float | None
    z: NDArray[np.float_] | float | None
    alpha: NDArray[np.float_] | float | None
    chi: NDArray[np.float_] | float | None
    theta: NDArray[np.float_] | float | None
    psi: NDArray[np.float_] | float | None


class ANALYZERINFO(TypedDict, total=False):
    """TypeDict for attrs.

    see analyzer_info in xarray_extensions.py
    """

    lens_mode: str | None
    lens_mode_name: str | None
    acquisition_mode: float
    pass_energy: float
    slit_shape: str
    slit_width: float
    slit_number: str | int | None
    lens_table: None
    analyzer_type: str
    mcp_voltage: float | None
    work_function: float | None
    #
    analyzer_radius: int | float
    analyzer: str
    analyzer_name: str
    parallel_deflectors: bool
    perpendicular_deflectors: bool


class PROBEINFO(TypedDict, total=False):
    """TypeDict for attrs.

    see probe_info in xarray_extensions.py
    """

    probe_wavelength: float | None
    probe_energy: float | None
    probe_fluence: float | None
    probe_pulse_energy: float | None
    probe_spot_size_x: float | None
    probe_spot_size_y: float | None
    probe_profile: None
    probe_linewidth: float
    probe_temporal_width: None
    probe_polarization: str | tuple[float | None, float | None]


class PUMPINFO(TypedDict, total=False):
    """TypeDict for attrs.

    see pump_info in xarray_extensions.py
    """

    pump_wavelength: float | None
    pump_energy: float | None
    pump_fluence: float | None
    pump_pulse_energy: float | None
    pump_spot_size_x: float | None
    pump_spot_size_y: float | None
    pump_profile: None
    pump_linewidth: float | None
    pump_temporal_width: float | None
    pump_polarization: str | tuple[float | None, float | None]


class BEAMLINEINFO(TypedDict, total=False):
    """TypeDict for attrs.

    see beamline_info in xarray_extensions.py
    """

    hv: float
    linewidth: float | None
    photon_polarization: tuple[float | None, float | None]
    undulation_info: Incomplete
    repetition_rate: float | None
    beam_current: float
    entrance_slit: float | None
    exit_slit: float | None
    monochrometer_info: dict[str, None | float]


class LIGHTSOURCE(PROBEINFO, PUMPINFO, BEAMLINEINFO, total=False):
    polarization: float | tuple[float | None, float | None] | str
    photon_flux: float | None
    photocurrent: float | None
    probe: None | float
    probe_detail: None


class SAMPLEINFO(TypedDict, total=False):
    id: int | str | None
    sample_name: str | None
    source: str | None
    reflectivity: float | None


class WORKSPACETYPE(TypedDict, total=False):
    path: str | Path
    name: str


class CURRENTCONTEXT(TypedDict, total=False):
    selected_components: float
    selected_indices: list[int]
    sum_data: Incomplete | None
    map_data: Incomplete | None
    selector: Incomplete | None
    integration_region: dict[Incomplete, Incomplete]
    original_data: Incomplete | None
    data: xr.DataArray | xr.Dataset
    widgets: list[Incomplete]
    points: list[Incomplete]
    rect_next: bool


class CONFIGTYPE(TypedDict, total=False):
    WORKSPACE: Required[WORKSPACETYPE]
    CURRENT_CONTEXT: CURRENTCONTEXT | None  # see widgets.py
    ENABLE_LOGGING: Required[bool]
    LOGGING_STARTED: Required[bool]
    LOGGING_FILE: Required[str | Path | None]


class SCANINFO(TypedDict, total=False):
    time: str
    data: str
    type: str | None
    spectrum_type: Literal["cut", "map"]
    experimenter: str | None
    sample: str | None


class EXPERIMENTALINFO(TypedDict, total=False):
    temperature: float | None
    temperature_cryotip: float | None
    pressure: float | None
    polarization: float | tuple[float | None, float | None] | str
    photon_flux: float | None
    photocurrent: float | None
    probe: None | float
    probe_detail: None


class DAQINFO(TypedDict, total=False):
    """TypeDict for attrs.

    see daq_info in xarray_extensions.py
    """

    daq_type: str | None
    region: str | None
    region_name: str | None
    center_energy: float | None
    prebinning: dict[str, float]
    trapezoidal_correction_strategy: Incomplete
    dither_settings: Incomplete
    sweep_setting: Incomplete
    frames_per_slice: int | None
    frame_duration: float | None


class SPECTROMETER(ANALYZERINFO, COORDINATES, total=False):
    name: str
    rad_per_pixel: float
    type: str  # noqa: A003
    is_slit_vertical: bool
    dof: list[str]
    scan_dof: list[str]
    mstar: float
    dof_type: dict[str, list[str]]
    length: float
    ##


class ARPESAttrs(TypedDict, total=False):
    pass
