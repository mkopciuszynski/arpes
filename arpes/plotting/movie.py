"""Utilities and an example of how to make an animated plot to export as a movie."""
from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import numpy as np
import xarray as xr
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import arpes.config
from arpes.plotting.utils import path_for_plot
from arpes.provenance import save_plot_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.collections import QuadMesh

    from arpes._typing import PColorMeshKwargs

__all__ = ("plot_movie",)


@save_plot_provenance
def plot_movie(
    data: xr.DataArray,
    time_dim: str = "delay",
    interval: float = 100,
    fig: Figure | None = None,
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | animation.FuncAnimation:
    """Make an animated plot of a 3D dataset using one dimension as "time".

    Args:
        data (xr.DataArray): ARPES data
        time_dim (str): dimension name for time
        interval: [TODO:description]
        fig: [TODO:description]
        ax: [TODO:description]
        out: [TODO:description]
        kwargs: [TODO:description]

    Raises:
        TypeError: [TODO:description]
    """
    if not isinstance(data, xr.DataArray):
        msg = "You must provide a DataArray"
        raise TypeError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)

    assert isinstance(arpes.config.SETTINGS, dict)
    cmap = arpes.config.SETTINGS.get("interactive", {}).get("palette", "viridis")
    vmax = data.max().item()
    vmin = data.min().item()

    if data.S.is_subtracted:
        cmap = "RdBu"
        vmax = np.max([np.abs(vmin), np.abs(vmax)])
        vmin = -vmax

    if "vmax" in kwargs:
        vmax = kwargs.pop("vmax")
    if "vmin" in kwargs:
        vmin = kwargs.pop("vmin")

    plot: QuadMesh = (
        data.mean(time_dim)
        .transpose()
        .plot(
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            **kwargs,
        )
    )

    def init() -> tuple[QuadMesh]:
        plot.set_array(np.asarray([]))
        return (plot,)

    animation_coords = data.coords[time_dim].values

    def animate(i: int) -> tuple[QuadMesh]:
        coordinate = animation_coords[i]
        data_for_plot = data.sel(**dict([[time_dim, coordinate]]))
        plot.set_array(data_for_plot.values.G.ravel())
        return (plot,)

    computed_interval = interval

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        repeat=500,
        frames=len(animation_coords),
        interval=computed_interval,
        blit=True,
    )

    animation_writer = animation.writers["ffmpeg"]
    writer = animation_writer(
        fps=1000 / computed_interval,
        metadata={"artist": "Me"},
        bitrate=1800,
    )

    if out:
        anim.save(str(path_for_plot(out)), writer=writer)
        return path_for_plot(out)

    return anim
