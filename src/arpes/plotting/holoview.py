"""Provides a hv based implementation of ImageTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv

#
from holoviews import DynamicMap, opts, streams

from arpes.constants import TWO_DIMENSION

if TYPE_CHECKING:
    import xarray as xr

hv.extension("bokeh")


def crosshair_view(dataarray: xr.DataArray) -> DynamicMap:
    """Show Crosshair view."""
    opts.defaults(opts.Curve(width=100))
    opts.defaults(opts.Text(color="#D3D3D3"))
    opts.defaults(opts.Image(active_tools=["pan"]))
    opts.defaults(opts.Image(default_tools=["save", "box_zoom", "reset"]))
    opts.defaults(opts.Image(cmap="viridis"))

    # Create two images
    assert dataarray.ndim == TWO_DIMENSION

    img = hv.Image(dataarray, kdims=list(dataarray.dims))
    argmax_coord = dataarray.G.argmax_coords()
    pointer = streams.PointerXY(
        x=argmax_coord[dataarray.dims[0]],
        y=argmax_coord[dataarray.dims[1]],
        source=img,
    )
    # Declare PointerX and dynamic VLine
    vline = hv.DynamicMap(
        lambda x, y: hv.VLine(x or -100),  # noqa: ARG005
        streams=[pointer],
    )
    hline = hv.DynamicMap(
        lambda x, y: hv.HLine(y or -100),  # noqa: ARG005
        streams=[pointer],
    )
    crosssection1 = img.apply.sample(**{str(dataarray.dims[0]): pointer.param.x})
    # By constructing a DynamicMap
    text = hv.DynamicMap(
        lambda x, y: hv.Text(
            x,
            y,
            f"{img[x, y]:.2f}",
            halign="left",
            valign="bottom",
        ),
        streams=[pointer],
    )

    return (img * hline * vline * text) << crosssection1
