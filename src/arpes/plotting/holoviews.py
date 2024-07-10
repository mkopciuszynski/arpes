"""Provides a holoviews based implementation of ImageTool."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, Unpack

import holoviews as hv
import numpy as np
from holoviews import DynamicMap, Image

from arpes.constants import TWO_DIMENSION

if TYPE_CHECKING:
    import xarray as xr

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

    2024/07/08: On Jupyterlab on safari, it may not work correctly.
    """
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)
    kwargs.setdefault("profile_view_height", 100)

    assert dataarray.ndim == TWO_DIMENSION
    max_coords = dataarray.G.argmax_coords()
    posx = hv.streams.PointerX(x=0)
    posy = hv.streams.PointerY(y=0)

    second_weakest_intensity = np.partition(np.unique(dataarray.values.flatten()), 1)[1]
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
    img: Image = hv.Image(
        (
            dataarray.coords[dataarray.dims[0]].values,
            dataarray.coords[dataarray.dims[1]].values,
            dataarray.values.T,
        ),
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
        lambda x: img.sample(
            **{str(dataarray.dims[0]): x if x else max_coords[dataarray.dims[0]]},
        ),
        streams=[posx],
    ).opts(
        ylim=plot_lim,
        width=kwargs["profile_view_height"],
        logx=kwargs["log"],
    )
    profile_y = hv.DynamicMap(
        lambda y: img.sample(
            **{str(dataarray.dims[1]): y if y else max_coords[dataarray.dims[1]]},
        ),
        streams=[posy],
    ).opts(
        ylim=plot_lim,
        height=kwargs["profile_view_height"],
        logy=kwargs["log"],
    )

    return img * hline * vline << profile_x << profile_y
