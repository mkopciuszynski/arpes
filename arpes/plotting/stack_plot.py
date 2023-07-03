"""Plotting routines for making the classic stacked line plots.

Think the album art for "Unknown Pleasures".
"""
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from arpes._typing import DataType, RGBColorType
from arpes.analysis.general import rebin
from arpes.plotting.tof import scatter_with_std
from arpes.plotting.utils import (
    colorbarmaps_for_axis,
    fancy_labels,
    generic_colorbarmap_for_data,
    label_for_dim,
    path_for_plot,
)
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

__all__ = (
    "stack_dispersion_plot",
    "flat_stack_plot",
    "offset_scatter_plot",
)


@save_plot_provenance
def offset_scatter_plot(
    data: DataType,
    name_to_plot: str = "",
    stack_axis: str = "",
    fermi_level=True,
    cbarmap=None,
    ax: Axes | None = None,
    out: str | Path = "",
    scale_coordinate=0.5,
    ylim=None,
    aux_errorbars=True,
    **kwargs,
) -> Path | tuple[Figure, Axes]:
    """Makes a stack plot (scatters version)."""
    assert isinstance(data, xr.Dataset)

    if not name_to_plot:
        var_names = [k for k in data.data_vars if "_std" not in k]  # => ["spectrum"]
        assert len(var_names) == 1
        name_to_plot = var_names[0]
        assert (name_to_plot + "_std") in data.data_vars
    two_dimensional = 2
    if len(data.data_vars[name_to_plot].dims) != two_dimensional:
        msg = "In order to produce a stack plot, data must be image-like."
        msg += "Passed data included dimensions:"
        msg += f" {data.data_vars[name_to_plot].dims}"
        raise ValueError(
            msg,
        )

    fig: Figure
    inset_ax = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=kwargs.get(
                "figsize",
                (
                    11,
                    5,
                ),
            ),
        )

    if inset_ax is None:
        inset_ax = inset_axes(ax, width="40%", height="5%", loc="upper left")

    if not stack_axis:
        stack_axis = data.data_vars[name_to_plot].dims[0]

    skip_colorbar = True
    if cbarmap is None:
        skip_colorbar = False
        try:
            cbarmap = colorbarmaps_for_axis[stack_axis]
        except:
            cbarmap = generic_colorbarmap_for_data(
                data.coords[stack_axis],
                ax=inset_ax,
                ticks=kwargs.get("ticks"),
            )

    cbar, cmap = cbarmap

    if not isinstance(cmap, matplotlib.colors.Colormap):
        # do our best
        try:
            cmap = cmap()
        except:
            # might still be fine
            pass

    # should be exactly two
    other_dim = [d for d in data.dims if d != stack_axis][0]

    if "eV" in data.dims and stack_axis != "eV" and fermi_level:
        ax.axhline(0, linestyle="--", color="red")
        ax.fill_betweenx([-1e6, 1e6], 0, 0.2, color="black", alpha=0.07)
        ax.set_ylim(ylim)

    # real plotting here
    for i, (coord, value) in enumerate(data.G.iterate_axis(stack_axis)):
        delta = data.G.stride(generic_dim_names=False)[other_dim]
        data_for = value.copy(deep=True)
        data_for.coords[other_dim] = data_for.coords[other_dim].copy(deep=True)
        data_for.coords[other_dim].values = data_for.coords[other_dim].values.copy()
        data_for.coords[other_dim].values -= i * delta * scale_coordinate / 10

        scatter_with_std(data_for, name_to_plot, ax=ax, color=cmap(coord[stack_axis]))

        if aux_errorbars:
            assert ylim is not None
            data_for = data_for.copy(deep=True)
            flattened = data_for.data_vars[name_to_plot].copy(deep=True)
            flattened.values = ylim[0] * np.ones(flattened.values.shape)
            data_for = data_for.assign(**{name_to_plot: flattened})
            scatter_with_std(
                data_for,
                name_to_plot,
                ax=ax,
                color=cmap(coord[stack_axis]),
                fmt="none",
            )

    ax.set_xlabel(other_dim)
    ax.set_ylabel(name_to_plot)
    fancy_labels(ax)

    try:
        if inset_ax and not skip_colorbar:
            inset_ax.set_xlabel(stack_axis, fontsize=16)

            fancy_labels(inset_ax)
            cbar(ax=inset_ax, **kwargs)
    except TypeError:
        # colorbar already rendered
        pass

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def flat_stack_plot(
    data: DataType,
    stack_axis=None,
    fermi_level=True,
    cbarmap=None,
    ax: Axes | None = None,
    mode="line",
    title: str = "",
    out: str | Path = "",
    transpose=False,
    **kwargs,
):
    """Generates a stack plot with all the lines distinguished by color rather than offset."""
    data_array = normalize_to_spectrum(data)
    two_dimensional = 2
    if len(data_array.dims) != two_dimensional:
        msg = "In order to produce a stack plot, data must be image-like."
        msg += f"Passed data included dimensions: {data_array.dims}"
        raise ValueError(
            msg,
        )

    fig: Figure
    inset_ax = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=kwargs.get(
                "figsize",
                (
                    7,
                    5,
                ),
            ),
        )
        inset_ax = inset_axes(ax, width="40%", height="5%", loc=1)

    if stack_axis is None:
        stack_axis = data_array.dims[0]

    skip_colorbar = True
    if cbarmap is None:
        skip_colorbar = False
        try:
            cbarmap = colorbarmaps_for_axis[stack_axis]
        except KeyError:
            cbarmap = generic_colorbarmap_for_data(
                data_array.coords[stack_axis],
                ax=inset_ax,
                ticks=kwargs.get("ticks"),
            )

    cbar, cmap = cbarmap

    # should be exactly two
    other_dim = [d for d in data_array.dims if d != stack_axis][0]
    other_coord = data_array.coords[other_dim]

    if not isinstance(cmap, matplotlib.colors.Colormap):
        # do our best
        try:
            cmap = cmap()
        except:
            # might still be fine
            pass

    if "eV" in data_array.dims and stack_axis != "eV" and fermi_level:
        if transpose:
            ax.axhline(0, color="red", alpha=0.8, linestyle="--", linewidth=1)
        else:
            ax.axvline(0, color="red", alpha=0.8, linestyle="--", linewidth=1)

    # meat of the plotting
    for coord_dict, marginal in list(data_array.G.iterate_axis(stack_axis)):
        if transpose:
            if mode == "line":
                ax.plot(
                    marginal.values,
                    marginal.coords[marginal.dims[0]].values,
                    color=cmap(coord_dict[stack_axis]),
                    **kwargs,
                )
            else:
                assert mode == "scatter"
                raise NotImplementedError
        else:
            if mode == "line":
                marginal.plot(ax=ax, color=cmap(coord_dict[stack_axis]), **kwargs)
            else:
                assert mode == "scatter"
                ax.scatter(*marginal.G.to_arrays(), color=cmap(coord_dict[stack_axis]), **kwargs)
                ax.set_xlabel(marginal.dims[0])

    ax.set_xlabel(label_for_dim(data_array, ax.get_xlabel()))
    ax.set_ylabel("Spectrum Intensity (arb).")
    ax.set_title(title, fontsize=14)
    ax.set_xlim([other_coord.min().item(), other_coord.max().item()])

    try:
        if inset_ax is not None and not skip_colorbar:
            inset_ax.set_xlabel(stack_axis, fontsize=16)
            fancy_labels(inset_ax)

            cbar(ax=inset_ax, **kwargs)
    except TypeError:
        # already rendered
        pass

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def stack_dispersion_plot(
    data: DataType,
    stack_axis: str = "",
    ax: Axes | None = None,
    title: str = "",
    out: str | Path = "",
    max_stacks: int = 100,
    *,
    transpose: bool = False,
    use_constant_correction=False,
    correction_side=None,
    color: RGBColorType | None = None,
    c: RGBColorType | None = None,
    label=None,
    shift=0,
    no_scatter=False,
    negate=False,
    s=1,
    scale_factor: float | None = None,
    linewidth: float = 1,
    palette=None,
    zero_offset: bool = False,
    uniform: bool = False,
    **kwargs,
) -> Path | tuple[Figure, Axes]:
    """Generates a stack plot with all the lines distinguished by offset rather than color.

    Args:
        data(DataType): ARPES data
        stack_axis(str): stack axis. e.g. "phi" , "eV", ...
        ax(Axes)
        title(str): Plot title, if not specified the attrs[description] (or S.scan_name) is used.
        out(str):
        transpose(bool)
        use_constant_correction(bool)
        correction_side()
        color()
        c()
        label()
        shift()
        no_scatter(bool)
        negate(bool)
        s()
        scale_factor(float)
        linewidth(float)
        pallette()
        zero_offset(bool)
        uniform(bool)
        **kwargs: pass to ax.plot (or ax.scatter)
    """
    data_arr = normalize_to_spectrum(data)
    assert isinstance(data_arr, xr.DataArray)
    if not stack_axis:
        stack_axis = data_arr.dims[0]

    other_axes = list(data_arr.dims)
    other_axes.remove(stack_axis)
    other_axis = other_axes[0]

    stack_coord = data_arr.coords[stack_axis]
    if len(stack_coord.values) > max_stacks:
        data_arr = rebin(
            data_arr,
            reduction=dict([[stack_axis, int(np.ceil(len(stack_coord.values) / max_stacks))]]),
        )

    fig: Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if not title:
        title = "{} Stack".format(data_arr.S.label.replace("_", " "))

    max_over_stacks = np.max(data_arr.values)

    cvalues = data_arr.coords[other_axis].values
    if scale_factor is None:
        maximum_deviation = -np.inf

        for _, marginal in data_arr.G.iterate_axis(stack_axis):
            marginal_values = -marginal.values if negate else marginal.values
            marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

            if use_constant_correction:
                true_ys = marginal_values - marginal_offset
            elif zero_offset:
                true_ys = marginal_values
            else:
                true_ys = marginal_values - np.linspace(
                    marginal_offset,
                    right_marginal_offset,
                    len(marginal_values),
                )

            maximum_deviation = np.max([maximum_deviation, *list(np.abs(true_ys))])

        scale_factor = 0.02 * (np.max(cvalues) - np.min(cvalues)) / maximum_deviation

    iteration_order = -1  # might need to fiddle with this in certain cases
    lim = [-np.inf, np.inf]
    labeled = False
    for i, (coord_dict, marginal) in enumerate(
        list(data_arr.G.iterate_axis(stack_axis))[::iteration_order],
    ):
        coord_value = coord_dict[stack_axis]

        xs = cvalues
        marginal_values = -marginal.values if negate else marginal.values
        marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

        if use_constant_correction:
            offset = right_marginal_offset if correction_side == "right" else marginal_offset
            true_ys = (marginal_values - offset) / max_over_stacks
            ys = scale_factor * true_ys + coord_value
        elif zero_offset:
            true_ys = marginal_values / max_over_stacks
            ys = scale_factor * true_ys + coord_value
        elif uniform:
            true_ys = marginal_values / max_over_stacks
            ys = scale_factor * true_ys + i
        else:
            true_ys = (
                marginal_values
                - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values))
            ) / max_over_stacks
            ys = scale_factor * true_ys + coord_value

        raw_colors = color or c or "black"

        if palette:
            if isinstance(palette, str):
                palette = cm.get_cmap(palette)
            raw_colors = palette(np.abs(true_ys / max_over_stacks))

        if transpose:
            xs, ys = ys, xs

        xs = xs - i * shift

        lim = [max(lim[0], np.min(xs)), min(lim[1], np.max(xs))]

        label_for = "_nolegend_"
        if not labeled:
            labeled = True
            label_for = label

        color_for_plot = raw_colors
        if callable(color_for_plot):
            color_for_plot = color_for_plot(coord_value)

        if isinstance(raw_colors, str | tuple) or no_scatter:
            ax.plot(xs, ys, linewidth=linewidth, color=color_for_plot, label=label_for, **kwargs)
        else:
            ax.scatter(xs, ys, color=color_for_plot, s=s, label=label_for, **kwargs)

    x_label = other_axis
    y_label = stack_axis

    if transpose:
        x_label, y_label = y_label, x_label

    ax.set_xlabel(label_for_dim(data_arr, x_label))
    ax.set_ylabel(label_for_dim(data_arr, y_label))

    if transpose:
        ax.set_ylim(lim)
    else:
        ax.set_xlim(lim)

    ax.set_title(title)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def overlapped_stack_dispersion_plot(
    data: DataType,
    stack_axis: str = "",
    ax: Axes | None = None,
    title: str = "",
    out: str | Path = "",
    max_stacks: int = 100,
    use_constant_correction=False,
    transpose=False,
    negate=False,
    s=1,
    scale_factor=None,
    linewidth: float = 1,
    palette=None,
    **kwargs,
):
    """Generate a Stack plot.

    Args:
        data(DataType): ARPES data
        stack_axis (str): axis for stacking (Default should be the S.spectrum.dims[0])
        ax(Axes | None): matplotlib Axes object
        title (str): Graph title
        out (str|Path) : Path for output graph view
        max_stacks(int): the number of maximum curves of spectrum
        use_constant_correction(bool):
        transpose(bool)
        negate(bool)
        s(int)
        scale_factor(float)
        linewidth=
    """
    data_arr = normalize_to_spectrum(data)
    if not stack_axis:
        stack_axis = data_arr.dims[0]

    other_axes = list(data_arr.dims)
    other_axes.remove(stack_axis)
    other_axis = other_axes[0]

    stack_coord = data_arr.coords[stack_axis]
    if len(stack_coord.values) > max_stacks:
        data_arr = rebin(
            data_arr,
            reduction=dict([[stack_axis, int(np.ceil(len(stack_coord.values) / max_stacks))]]),
        )

    fig: Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if not title:
        title = "{} Stack".format(data_arr.S.label.replace("_", " "))

    max_over_stacks = np.max(data_arr.values)

    cvalues = data_arr.coords[other_axis].values
    if scale_factor is None:
        maximum_deviation = -np.inf

        for _, marginal in data_arr.G.iterate_axis(stack_axis):
            marginal_values = -marginal.values if negate else marginal.values
            marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

            if use_constant_correction:
                true_ys = marginal_values - marginal_offset
            else:
                true_ys = marginal_values - np.linspace(
                    marginal_offset,
                    right_marginal_offset,
                    len(marginal_values),
                )

            maximum_deviation = np.max([maximum_deviation, *list(np.abs(true_ys))])

        scale_factor = 0.02 * (np.max(cvalues) - np.min(cvalues)) / maximum_deviation

    iteration_order = -1  # might need to fiddle with this in certain cases
    for coord_dict, marginal in list(data_arr.G.iterate_axis(stack_axis))[::iteration_order]:
        coord_value = coord_dict[stack_axis]

        xs = cvalues
        marginal_values = -marginal.values if negate else marginal.values
        marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

        if use_constant_correction:
            true_ys = (marginal_values - marginal_offset) / max_over_stacks
            ys = scale_factor * true_ys + coord_value
        else:
            true_ys = (
                marginal_values
                - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values))
            ) / max_over_stacks
            ys = scale_factor * true_ys + coord_value

        raw_colors = "black"
        if palette:
            if isinstance(palette, str):
                palette = cm.get_cmap(palette)
            raw_colors = palette(np.abs(true_ys / max_over_stacks))

        if transpose:
            xs, ys = ys, xs

        if isinstance(raw_colors, str):
            plt.plot(xs, ys, linewidth=linewidth, color=raw_colors, **kwargs)
        else:
            plt.scatter(xs, ys, color=raw_colors, s=s, **kwargs)

    x_label = other_axis
    y_label = stack_axis

    if transpose:
        x_label, y_label = y_label, x_label

    ax.set_xlabel(label_for_dim(data_arr, x_label))
    ax.set_ylabel(label_for_dim(data_arr, y_label))

    ax.set_title(title)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()

    return fig, ax
