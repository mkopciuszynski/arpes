"""Plotting utilities related to density of states plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Unpack

import xarray as xr
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from arpes.analysis.xps import approximate_core_levels
from arpes.constants import TWO_DIMENSION
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import path_for_plot, savefig

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from arpes._typing import MPLPlotKwargs, QuadmeshParam

__all__ = (
    "plot_core_levels",
    "plot_dos",
)


@save_plot_provenance
def plot_core_levels(  # noqa: PLR0913
    data: xr.DataArray,
    ax: Axes | None = None,
    out: str | Path = "",
    core_levels: list[float] | None = None,
    binning: int = 3,
    promenance: int = 5,
    figsize: tuple[float, float] = (11, 5),
    **kwargs: Unpack[MPLPlotKwargs],
) -> Path | tuple[Figure, Axes]:
    """Plots an XPS curve and approximate core level locations."""
    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)
    data.S.plot(ax=ax, **kwargs)

    if core_levels is None:
        core_levels = approximate_core_levels(data, binning=binning, promenance=promenance)
    assert core_levels is not None
    for core_level in core_levels:
        ax.axvline(core_level, ymin=0.1, ymax=0.25, color="red", ls="-")

    if out:
        savefig(str(out), dpi=400)
        return path_for_plot(out)
    return fig, ax


@save_plot_provenance
def plot_dos(
    data: xr.DataArray,
    out: str | Path = "",
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    **kwargs: Unpack[QuadmeshParam],
) -> Path | tuple[Figure, tuple[Axes, Axes]]:
    """Plots the density of states next to the original spectr spectra.

    Todo:
        1. Add kwargs.
        2. Add orientation args.

    cf: https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_inset_locator.html

    Args:
        data: ARPES data to plot.
        out (str | Path): Path to the figure.
        orientation (Literal["horizontal", "vetical"]): Orientation of the figures.
        kwargs: pass to the original data.

    Returns: fig, tuple[Axes, Axes]
        Figure object and the Axes images of spectra and the line profile of the spectrum.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data, xr.DataArray)
    assert data.ndim == TWO_DIMENSION
    dos = data.S.sum_other(["eV"])
    kwargs.setdefault(
        "norm",
        Normalize(vmin=data.min().item(), vmax=data.max().item()),
    )
    if orientation.startswith("h"):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[5.5, 1])
    else:
        data = data.transpose()
        fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5.5, 1])
    fig.subplots_adjust(hspace=0.00, wspace=0.00)
    ax0 = fig.add_subplot(gs[0])
    data.S.plot(ax=ax0, add_labels=False, add_colorbar=False, **kwargs)

    if orientation.startswith("h"):
        ax1: Axes = fig.add_subplot(gs[1], sharex=ax0)
        axins = inset_axes(
            ax0,
            width="2%",
            height="90%",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.05, 1, 1),
            bbox_transform=ax0.transAxes,
        )
        Colorbar(
            ax=axins,
            orientation="vertical",
            norm=kwargs.get("norm"),
        )
        plt.setp(ax0.get_xticklabels(), visible=False)
        dos.S.plot(ax=ax1, _labels=False)
        ax1.set_xlabel(str(data.dims[1]))
    else:  # Vertical orientation.
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        axins = inset_axes(
            ax0,
            height="2%",
            width="90%",
            loc="lower left",
            bbox_to_anchor=(0.01, 1.05, 1, 1),
            bbox_transform=ax0.transAxes,
        )
        Colorbar(
            ax=axins,
            orientation="horizontal",
            norm=kwargs.get("norm"),
        )
        plt.setp(ax1.get_yticklabels(), visible=False)
        dos.S.plot(ax=ax1, _labels=False, y="eV")
        ax0.set_xlabel(str(data.dims[1]))
        ax1.set_xlabel("Intensity")
    ax0.set_ylabel(str(data.dims[0]))
    assert "norm" in kwargs
    assert isinstance(kwargs["norm"], Normalize | None)

    if out:
        savefig(str(out), dpi=400)
        return path_for_plot(out)
    return fig, (ax0, ax1)
