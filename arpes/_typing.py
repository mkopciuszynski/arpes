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
from typing import TYPE_CHECKING, Literal, Required, TypeAlias, TypedDict, TypeVar

import xarray as xr

# from Matplotlib 3.8dev
# After 3.8 release, the two lines below should be removed.
##

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import matplotlib as mpl
    import numpy as np
    from _typeshed import Incomplete
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import Event
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.transforms import BboxBase, Transform
    from matplotlib.typing import (
        CapStyleType,
        ColorType,
        DrawStyleType,
        FillStyleType,
        JoinStyleType,
        LineStyleType,
        MarkerType,
        MarkEveryType,
    )
    from numpy.typing import ArrayLike, NDArray

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
NormalizableDataType: TypeAlias = DataType | str | uuid.UUID

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


class MPLPlotKwargs(TypedDict, total=False):
    scalex: bool
    scaley: bool

    agg_filter: Callable[[NDArray[np.float_], int], tuple[NDArray[np.float_], int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    aa: bool
    clip_box: BboxBase | None
    clip_on: bool
    # clip_path: Path | None color: ColorType
    c: ColorType
    dash_capstyle: CapStyleType
    dash_joinstyle: JoinStyleType
    dashes: Sequence[float | None]
    data: NDArray[np.float_]
    drawstyle: DrawStyleType
    figure: Figure
    fillstyle: FillStyleType
    gapcolor: ColorType | None
    gid: str
    in_layout: bool
    label: str
    linestyle: LineStyleType
    ls: LineStyleType
    linewidth: float
    lw: float
    marker: MarkerType
    markeredgecolor: ColorType
    mec: ColorType
    markeredgewidth: float
    mew: float
    markerfacecolor: ColorType
    mfc: ColorType
    markerfacecoloralt: ColorType
    mfcalt: ColorType
    markersize: float
    ms: float
    markevery: MarkEveryType
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    picker: float | Callable[[Artist, Event], tuple[bool, dict]]
    pickradius: float
    rasterized: bool
    sketch_params: tuple[float, float, float]
    scale: float
    length: float
    randomness: float
    snap: bool | None
    solid_capstyle: CapStyleType
    solid_joinstyle: JoinStyleType
    url: str
    visible: bool
    xdata: NDArray[np.float_]
    ydata: NDArray[np.float_]
    zorder: float


class ColorbarParam(TypedDict, total=False):
    alpha: float
    orientation: None | Literal["vertical", "horizontal"]
    ticklocation: Literal["auto", "right", "top", "bottom"]
    drawedge: bool
    extend: Literal["neither", "both", "min", "max"]
    extendfrac: None | Literal["auto"] | float | tuple[float, float] | list[float]
    spacing: Literal["uniform", "proportional"]
    ticks: None | list[float]
    format: str | None
    drawedges: bool
    label: str
    boundaries: None | Sequence[float]
    values: None | Sequence[float]
    location: None | Literal["left", "right", "top", "bottom"]
    cmap: Colormap
    norm: Normalize


class MPLTextParam(TypedDict, total=False):
    agg_filter: Callable[[NDArray[np.float_], int], tuple[NDArray[np.float_], int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    backgroundcolor: ColorType
    color: ColorType
    c: ColorType
    figure: Figure
    fontfamily: str
    family: str
    fontname: str
    fontproperties: str | Path
    font: str | Path
    font_properties: str | Path
    fontsize: (
        float
        | Literal[
            "xx-small",
            "x-small",
            "small",
            "medium",
            "large",
            "x-large",
            "xx-large",
        ]
    )
    size: (
        float
        | Literal[
            "xx-small",
            "x-small",
            "small",
            "medium",
            "large",
            "x-large",
            "xx-large",
        ]
    )
    fontstretch: (
        float
        | Literal[
            "ultra-condensed",
            "extra-condensed",
            "condensed",
            "semi-condensed",
            "normal",
            "semi-expanded",
            "expanded",
            "extra-expanded",
            "ultra-expanded",
        ]
    )
    stretch: (
        float
        | Literal[
            "ultra-condensed",
            "extra-condensed",
            "condensed",
            "semi-condensed",
            "normal",
            "semi-expanded",
            "expanded",
            "extra-expanded",
            "ultra-expanded",
        ]
    )
    fontstyle: Literal["normal", "italic", "oblique"]
    style: Literal["normal", "italic", "oblique"]
    fontvariant: Literal["normal", "small-caps"]
    variant: Literal["normal", "small-caps"]
    fontweight: (
        float
        | Literal[
            "ultralight",
            "light",
            "normal",
            "regular",
            "book",
            "medium",
            "roman",
            "semibold",
            "demibold",
            "demi",
            "bold",
            "heavy",
            "extra bold",
            "black",
        ]
    )
    weight: (
        float
        | Literal[
            "ultralight",
            "light",
            "normal",
            "regular",
            "book",
            "medium",
            "roman",
            "semibold",
            "demibold",
            "demi",
            "bold",
            "heavy",
            "extra bold",
            "black",
        ]
    )
    gid: str
    horizontalalignment: Literal["left", "center", "right"]
    ha: Literal["left", "center", "right"]
    in_layout: bool
    label: str
    linespacing: float
    math_fontfamily: str
    mouseover: bool
    multialignment: Literal["left", "right", "center"]
    ma: Literal["left", "right", "center"]
    parse_math: bool
    path_effects: list[AbstractPathEffect]
    picker: None | bool | float | Callable
    position: tuple[float, float]
    rasterized: bool
    rotation: float | Literal["vertical", "horizontal"]
    rotation_mode: Literal[None, "default", "anchor"]
    sketch_params: tuple[float, float, float]
    scale: float
    length: float
    randomness: float
    snap: bool | None
    text: str
    transform: Transform
    transform_rotates_text: bool
    url: str
    usetex: bool | None
    verticalalignment: Literal["bottom", "baseline", "center", "center_baseline", "top"]
    va: Literal["bottom", "baseline", "center", "center_baseline", "top"]
    visible: bool
    wrap: bool
    zorder: float


class PLTSubplotParam(TypedDict, total=False):
    sharex: bool | Literal["none", "all", "row", "col"]
    sharey: bool | Literal["none", "all", "row", "col"]
    squeeze: bool
    width_ratios: Sequence[float] | None
    height_ratios: Sequence[float] | None
    subplot_kw: dict
    gridspec_kw: dict


class AxesImageParam(TypedDict, total=False):
    ax: Axes
    cmap: str | Colormap
    norm: str | Normalize
    interpolation: Literal[
        "none",
        "antialiased",
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
        "blackman",
    ]
    interpolation_stage: Literal["data", "rgba"]
    origin: Literal["upper", "lower"]
    extent: tuple[float, float, float, float]
    filternorm: bool
    filterrad: float
    resample: bool


class IMshowParam(AxesImageParam, total=False):
    aspect: Literal["equal", "auto"] | float | None
    alpha: float
    vmin: float
    vmax: float
    url: str


class QuadmeshParam(TypedDict, total=False):
    agg_filter: Callable[..., tuple[NDArray[np.ndindex], float, float]]
    alpha: float
    animated: bool
    antialiased: bool | list[bool]
    aa: bool | list[bool]
    antialiaseds: bool | list[bool]
    array: ArrayLike
    capstyle: CapStyleType
    clim: tuple[float, float]
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Patch | (mpl.Path, Transform) | None
    cmap: Colormap | str | None
    color: ColorType
    edgecolor: ColorType
    ec: ColorType
    edgecolors: ColorType
    facecolor: ColorType
    facecolors: ColorType
    fc: ColorType
    figure: Figure
    gid: str
    hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    in_layout: bool
    joinstyle: JoinStyleType
    label: str
    linestyle: LineStyleType
    dashes: LineStyleType
    linestyles: LineStyleType
    ls: LineStyleType
    linewidth: float | list[float]
    linewidths: float | list[float]
    lw: float | list[float]
    mouseover: bool
    norm: Normalize | str | None
    offset_transform: Transform
    transOffset: Transform
    offsets: ArrayLike
    path_effects: list[AbstractPathEffect]
    picker: None | bool | float
    pickradius: float
    rasterized: bool
    sketch_params: tuple[float, float, float]
    scale: float
    randomness: float
    snap: bool | None
    transform: Transform
    url: str
    urls: list[str | None]
    visible: bool
    zorder: float


class PColorMeshKwargs(QuadmeshParam, total=False):
    vim: float
    vmax: float
    shading: Literal["flat", "nearest", "gouraud", "auto"]
    snap: bool
    rasterized: bool
