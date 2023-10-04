"""For plotting band locations."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from arpes.provenance import save_plot_provenance

from .utils import label_for_colorbar, path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from _typeshed import Incomplete
    from build.lib.arpes.typing import DataType
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize

__all__ = ("plot_with_bands",)


@save_plot_provenance
def plot_with_bands(
    data: DataType,
    bands,
    title: str = "",
    ax: Axes | None = None,
    norm: Normalize | None = None,
    out: str | Path = "",
    **kwargs: Incomplete,
) -> Path | Axes:  # <== CHECKME the type may be NDArray[np.object_]
    """Makes a dispersion plot with bands overlaid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if not title:
        title = data.S.label.replace("_", " ")

    mesh = data.plot(norm=norm, ax=ax, **kwargs)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap("Blues")

    for band in bands:
        plt.scatter(band.center.values, band.coords[band.dims[0]].values)

    if out:
        filename = path_for_plot(out)
        plt.savefig(filename)
        return filename
    plt.show()
    return ax
