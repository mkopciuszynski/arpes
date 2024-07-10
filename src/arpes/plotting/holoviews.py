"""Provides a holoviews based implementation of ImageTool."""

from __future__ import annotations

from typing import TypedDict, Unpack

import holoviews as hv
import numpy as np
import xarray as xr
from holoviews import DynamicMap, Image

from arpes.constants import TWO_DIMENSION
from arpes.utilities.normalize import normalize_to_spectrum

hv.extension("bokeh")


class CrosshairViewParam(TypedDict):
    """Kwargs for crosshair_view."""

    width: int
    height: int
    cmap: str
    log: bool
    profile_view_height: int


def crosshair_view(dataarray: xr.DataArray, **kwargs: Unpack[CrosshairViewParam]) -> None:
    """Show Crosshair view.

    Todo:
    There are some issues.

    * 2024/07/08: On Jupyterlab on safari, it may not work correctly.
    * 2024/07/09: Some xr.DataArray data (especially the output of the convert_to_kspace() cannot be
      handled correctly.  (https://github.com/holoviz/holoviews/issues/6317)
    * 2024/07/10: Incompatibility between bokeh and matplotlib about which is "x-" axis about
      plotting xarray data.

    """
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)
    kwargs.setdefault("profile_view_height", 100)

    assert dataarray.ndim == TWO_DIMENSION
    max_coords = dataarray.G.argmax_coords()
    posx = hv.streams.PointerX(x=max_coords[dataarray.dims[0]])
    posy = hv.streams.PointerY(y=max_coords[dataarray.dims[1]])

    second_weakest_intensity = np.partition(np.unique(dataarray.values.flatten()), 1)[1]
    dataarray = (
        dataarray if isinstance(dataarray, xr.DataArray) else normalize_to_spectrum(dataarray)
    )
    plot_lim: tuple[None | np.float64, np.float64] = (
        (second_weakest_intensity * 0.1, dataarray.max().item() * 10)
        if kwargs["log"]
        else (None, dataarray.max().item() * 1.1)
    )
    vline: DynamicMap = hv.DynamicMap(
        lambda x: hv.VLine(x=x or max_coords[dataarray.dims[0]]),
        streams=[posx],
    )
    hline: DynamicMap = hv.DynamicMap(
        lambda y: hv.HLine(y=y or max_coords[dataarray.dims[1]]),
        streams=[posy],
    )
    # Memo: (ad-hoc fix) to avoid the problem concerning https://github.com/holoviz/holoviews/issues/6317
    img: Image = hv.Image(
        (
            dataarray.coords[dataarray.dims[1]].values,
            dataarray.coords[dataarray.dims[0]].values,
            dataarray.values,
        ),
        kdims=list(dataarray.dims),
        vdims=["spectrum"],
    ).opts(
        width=kwargs["width"],
        height=kwargs["height"],
        logz=kwargs["log"],
        cmap=kwargs["cmap"],
        clim=plot_lim,
        active_tools=["box_zoom"],
        default_tools=["save", "box_zoom", "reset", "hover"],
    )
    profile_x = hv.DynamicMap(
        lambda x: img.sample(**{str(dataarray.dims[0]): x if x else max_coords[dataarray.dims[0]]}),
        streams=[posx],
    ).opts(ylim=plot_lim, width=kwargs["profile_view_height"], logx=kwargs["log"])
    profile_y = hv.DynamicMap(
        lambda y: img.sample(**{str(dataarray.dims[1]): y if y else max_coords[dataarray.dims[1]]}),
        streams=[posy],
    ).opts(ylim=plot_lim, height=kwargs["profile_view_height"], logy=kwargs["log"])

    return img * hline * vline << profile_x << profile_y
