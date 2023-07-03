"""Utilities and an example of how to make an animated plot to export as a movie."""
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import arpes.config
from arpes.plotting.utils import path_for_plot
from arpes.provenance import save_plot_provenance

__all__ = ("plot_movie",)


@save_plot_provenance
def plot_movie(
    data: xr.DataArray,
    time_dim,
    interval: float = 100,
    fig=None,
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs,
):
    """Make an animated plot of a 3D dataset using one dimension as "time"."""
    if not isinstance(data, xr.DataArray):
        msg = "You must provide a DataArray"
        raise TypeError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    assert isinstance(ax, Axes)

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

    plot = data.mean(time_dim).transpose().plot(vmax=vmax, vmin=vmin, cmap=cmap, **kwargs)

    def init():
        plot.set_array(np.asarray([]))
        return (plot,)

    animation_coords = data.coords[time_dim].values

    def animate(i):
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

    writer = animation.writers["ffmpeg"](
        fps=1000 / computed_interval,
        metadata={"artist": "Me"},
        bitrate=1800,
    )

    if out:
        anim.save(path_for_plot(out), writer=writer)
        return path_for_plot(out)

    return anim
