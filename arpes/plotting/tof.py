"""Plotting routines for count-like data from a ToF or delay line.

This module is a bit of a misnomer, in that it also applies perfectly well to data collected by a
delay line on a hemisphere, the important point is that the data in any given channel should
correspond to the true number of electrons that arrived in that channel.

Plotting routines here are ones that include statistical errorbars. Generally for datasets in
PyARPES, an xr.Dataset will hold the standard deviation data for a given variable on
`{var_name}_std`.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arpes._typing import DataType
from arpes.plotting.utils import path_for_plot
from arpes.provenance import save_plot_provenance

__all__ = (
    "plot_with_std",
    "scatter_with_std",
)


@save_plot_provenance
def plot_with_std(
    data: DataType,
    name_to_plot: str = "",
    ax: plt.Axes | None = None,
    out: str | Path = "",
    **kwargs,
) -> Path | tuple[plt.Figure, plt.Axes]:
    """Makes a fill-between line plot with error bars from associated statistical errors."""
    if not name_to_plot:
        var_names = [k for k in data.data_vars if "_std" not in k]
        assert len(var_names) == 1
        name_to_plot = var_names[0]
        assert (name_to_plot + "_std") in data.data_vars

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(
            figsize=kwargs.pop(
                "figsize",
                (
                    7,
                    5,
                ),
            ),
        )

    data.data_vars[name_to_plot].plot(ax=ax, **kwargs)
    x, y = data.data_vars[name_to_plot].G.to_arrays()

    std = data.data_vars[name_to_plot + "_std"].values
    ax.fill_between(x, y - std, y + std, alpha=0.3, **kwargs)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    ax.set_xlim([np.min(x), np.max(x)])

    return fig, ax


@save_plot_provenance
def scatter_with_std(
    data: DataType,
    name_to_plot: str = "",
    ax: plt.Axes | None = None,
    fmt: str = "o",
    out: str | Path = "",
    **kwargs,
) -> Path | tuple[plt.Figure, plt.Axes]:
    """Makes a scatter plot of data with error bars generated from associated statistical errors."""
    if not name_to_plot:
        var_names = [k for k in data.data_vars if "_std" not in k]
        assert len(var_names) == 1
        name_to_plot = var_names[0]
        assert (name_to_plot + "_std") in data.data_vars

    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=kwargs.pop(
                "figsize",
                (
                    7,
                    5,
                ),
            ),
        )

    x, y = data.data_vars[name_to_plot].G.to_arrays()

    std = data.data_vars[name_to_plot + "_std"].values
    ax.errorbar(x, y, yerr=std, fmt=fmt, markeredgecolor="black", **kwargs)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    ax.set_xlim([np.min(x), np.max(x)])

    return fig, ax
