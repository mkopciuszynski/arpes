"""Provides a holoviews based implementation of ImageTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
from holoviews import DynamicMap, Image, opts

from arpes.constants import TWO_DIMENSION

if TYPE_CHECKING:
    import xarray as xr

hv.extension("bokeh")


def crosshair_view(dataarray: xr.DataArray) -> None:
    """Show Crosshair view.

    2024/07/08: On Jupyterlab on safari, it may not work correctly.
    """
    opts.defaults(opts.Text(color="#D3D3D3"))
    opts.defaults(opts.Image(active_tools=["pan"]))
    opts.defaults(opts.Image(default_tools=["save", "box_zoom", "reset", "hover"]))
    opts.defaults(opts.Image(cmap="viridis"))
    profile_view_height = 100
    assert dataarray.ndim == TWO_DIMENSION
    max_coords = dataarray.G.argmax_coords()
    # Create two images
    posx = hv.streams.PointerX(x=0)
    posy = hv.streams.PointerY(y=0)
    vline: DynamicMap = hv.DynamicMap(
        lambda x: hv.VLine(x=x or max_coords[dataarray.dims[0]]),
        streams=[posx],
    )
    hline: DynamicMap = hv.DynamicMap(
        lambda y: hv.HLine(y=y or max_coords[dataarray.dims[1]]),
        streams=[posy],
    )
    img: Image = hv.Image(dataarray, kdims=list(dataarray.dims))
    profile_x = hv.DynamicMap(
        lambda x: img.sample(**{str(dataarray.dims[0]): x if x else 0}),
        streams=[posx],
    ).opts(ylim=(None, dataarray.max().item() * 1.1), width=profile_view_height)
    profile_y = hv.DynamicMap(
        lambda y: img.sample(**{str(dataarray.dims[1]): y if y else 0}),
        streams=[posy],
    ).opts(ylim=(None, dataarray.max().item() * 1.1), height=profile_view_height)

    return img * hline * vline << profile_x << profile_y
