"""Annotations onto plots for experimental conditions or locations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import numpy as np
from matplotlib.axes import Axes3D

from arpes.plotting.utils import name_for_dim, unit_for_dim
from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import Incomplete
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from arpes._typing import DataType, MPLTextParam

__all__ = (
    "annotate_cuts",
    "annotate_point",
    "annotate_experimental_conditions",
)

TWODimensional = 2


def annotate_experimental_conditions(
    ax: Axes,
    data: DataType,
    desc: list[str | float] | float | str,
    *,
    show: bool = False,
    orientation: str = "top",
    **kwargs: Unpack[MPLTextParam],
) -> None:
    """Renders information about the experimental conditions onto a set of axes.

    Also adjust the axes limits and hides the axes.

    data should be the dataset described, and desc should be one of

    'temp',
    'photon',
    'photon polarization',
    'polarization',

    or a number to act as a spacer in units of the axis coordinates

    or a list of such items.
    """
    if isinstance(desc, str | int | float):
        desc = [desc]

    ax.grid(visible=False)
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=100)
    if not show:
        ax.set_axis_off()
        ax.patch.set_alpha(0)

    delta = -1
    current = 100.0
    if orientation == "bottom":
        delta = 1
        current = 0

    fontsize = kwargs.get("fontsize", 16)
    delta = fontsize * delta

    conditions = data.S.experimental_conditions

    def render_polarization(c: dict[str, Incomplete]) -> str:
        pol = c["polarization"]
        if pol in ["lc", "rc"]:
            return "\\textbf{" + pol.upper() + "}"

        symbol_pol = {
            "s": "",
            "p": "",
            "s-p": "",
            "p-s": "",
        }

        prefix = ""
        if pol in ["s-p", "p-s"]:
            prefix = "\\textbf{Linear Dichroism, }"

        symbol = symbol_pol[pol]
        if symbol:
            return prefix + "$" + symbol + "$/\\textbf{" + pol + "}"

        return prefix + "\\textbf{" + pol + "}"

    def render_photon(c: dict[str, float]) -> str:
        return "\\textbf{" + str(c["hv"]) + " eV"

    renderers = {
        "temp": lambda c: "\\textbf{T = " + "{:.3g}".format(c["temp"]) + " K}",
        "photon": render_photon,
        "photon polarization": lambda c: render_photon(c) + ", " + render_polarization(c),
        "polarization": render_polarization,
    }

    for item in desc:
        if isinstance(item, float):
            current += item + delta
            continue

        item = item.replace("_", " ").lower()

        ax.text(0, current, renderers[item](conditions), **kwargs)
        current += delta


def annotate_cuts(
    ax: Axes,
    data: DataType,
    plotted_axes: NDArray[np.object_],
    *,
    include_text_labels: bool = False,
    **kwargs: Incomplete,
) -> None:
    """Annotates a cut location onto a plot.

    Example:
        >>> annotate_cuts(ax, conv, ['kz', 'ky'], hv=80)  # doctest: +SKIP

    Args:
        ax: The axes to plot onto
        data: The original data
        plotted_axes: The dimension names which were plotted
        include_text_labels: Whether to include text labels
        kwargs: Defines the coordinates of the cut location
    """
    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    assert len(plotted_axes) == TWODimensional

    for k, v in kwargs.items():
        if not isinstance(v, tuple | list | np.ndarray):
            v = [v]

        selected = converted_coordinates.sel(**dict([[k, v]]), method="nearest")

        for coords_dict, obj in selected.G.iterate_axis(k):
            css = [obj[d].values for d in plotted_axes]
            ax.plot(*css, color="red", ls="--", linewidth=1, dashes=(5, 5))

            if include_text_labels:
                idx = np.argmin(css[1])

                ax.text(
                    css[0][idx] + 0.05,
                    css[1][idx],
                    f"{name_for_dim(k)} = {coords_dict[k].item()} {unit_for_dim(k)}",
                    color="red",
                    size="medium",
                )


def annotate_point(
    ax: Axes | Axes3D,
    location: Sequence[float],
    label: str,
    delta: tuple[float, ...] = (),
    **kwargs: Unpack[MPLTextParam],
) -> None:
    """Annotates a point or high symmetry location into a plot."""
    label = {
        "G": "$\\Gamma$",
        "X": r"\textbf{X}",
        "Y": r"\textbf{Y}",
        "K": r"\textbf{K}",
        "M": r"\textbf{M}",
    }.get(label, label)

    if not delta:
        delta = (
            -0.05,
            0.05,
        )
    if "color" not in kwargs:
        kwargs["color"] = "red"

    if len(delta) == TWODimensional:
        dx, dy = tuple(delta)
        pos_x, pos_y = tuple(location)
        ax.plot([pos_x], [pos_y], "o", c=kwargs["color"])
        ax.text(pos_x + dx, pos_y + dy, label, **kwargs)
    else:
        assert isinstance(ax, Axes3D)
        dx, dy, dz = tuple(delta)
        pos_x, pos_y, pos_z = tuple(location)
        ax.plot([pos_x], [pos_y], [pos_z], "o", c=kwargs["color"])
        ax.text(pos_x + dx, pos_y + dy, pos_z + dz, label, **kwargs)
