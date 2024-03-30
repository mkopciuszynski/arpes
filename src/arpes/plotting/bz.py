"""Utilities related to plotting Brillouin zones and data onto them."""

from __future__ import annotations

import itertools
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, TypeAlias

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from arpes.analysis.mask import apply_mask_to_coords
from arpes.constants import TWO_DIMENSION
from arpes.utilities.bz import build_2dbz_poly, hex_cell_2d, process_kpath
from arpes.utilities.bz_spec import A_GRAPHENE, A_WS2, A_WSe2
from arpes.utilities.geometry import polyhedron_intersect_plane

from .utils import path_for_plot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from _typeshed import Incomplete
    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike, NDArray

__all__ = (
    "annotate_special_paths",
    "bz2d_plot",
    "bz3d_plot",
    "bz_plot",
    "plot_data_to_bz",
    "plot_data_to_bz2d",
    "plot_data_to_bz3d",
    "plot_plane_to_bz",
    "bz2d_segments",
    "overplot_standard",
)

overplot_library: dict[str, Callable[..., dict[str, list[list[float]]]]] = {
    "graphene": lambda: {"cell": hex_cell_2d(A_GRAPHENE)},
    "ws2": lambda: {"cell": hex_cell_2d(A_WS2)},
    "wwe2": lambda: {"cell": hex_cell_2d(A_WSe2)},
}


LOGLEVEL = (DEBUG, INFO)[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def segments_standard(
    name: str = "graphene",
    rotate_rad: float = 0.0,
) -> tuple[list[NDArray[np.float_]], list[NDArray[np.float_]]]:
    name = name.lower()
    specification: dict[str, list[list[float]]] = overplot_library[name]()
    transformations = []
    if rotate_rad:
        transformations = [Rotation.from_rotvec([0, 0, rotate_rad])]

    return bz2d_segments(specification["cell"], transformations)


def overplot_standard(
    name: str = "graphene",
    repeat: tuple[int, int, int] | tuple[int, int] | None = None,
    rotate: float = 0,
) -> Callable[[Axes], Axes]:
    """A higher order function to plot a Brillouin zone over a plot."""
    specification = overplot_library[name]()
    transformations = []

    if rotate:
        transformations = [Rotation.from_rotvec([0, 0, rotate])]

    def overplot_the_bz(ax: Axes) -> Axes:
        return bz_plot(
            cell=specification["cell"],
            linewidth=2,
            ax=ax,
            paths=[],
            repeat=repeat,
            set_equal_aspect=False,
            hide_ax=False,
            transformations=transformations,
            zorder=5,
            linestyle="-",
        )

    return overplot_the_bz


class Translation:
    """Base translation class, meant to provide some extension over rotations.

    Rotations are available from `scipy.spatial.transform.Rotation`.
    """

    translation_vector = None

    def __init__(self, translation_vector: ArrayLike) -> None:
        self.translation_vector = np.asarray(translation_vector)

    def apply(self, vectors: ArrayLike, *, inverse: bool = False) -> NDArray[np.float_]:
        """Applies the translation to a set of vectors.

        If this transform is D-dimensional (for D=2,3) and is applied to a different
        dimensional set of vectors, a ValueError will be thrown due to the dimension
        mismatch.

        An inverse flag is available in order to apply the inverse coordinate transform.
        Up to numerical accuracy,

        ```
        self.apply(self.apply(vectors), inverse=True) == vectors
        ```

        Args:
            vectors: array_like with shape (2 or 3,) or (N, 2 or 3)
            inverse: Applies the inverse coordinate transform instead
        """
        vectors = np.asarray(vectors)

        if vectors.ndim > TWO_DIMENSION or vectors.shape[-1] not in {2, 3}:
            msg = "Expected a 2D or 3D vector (2 or 3,)"
            msg += f" of list of vectors (N, 2 or 3,), instead receivied: {vectors.shape}"
            raise ValueError(
                msg,
            )

        single_vector = False
        if vectors.ndim == 1:
            single_vector = True
            vectors = vectors[None, :]  # expand dims

        result = vectors - self.translation_vector if inverse else vectors + self.translation_vector

        return result if not single_vector else result[0]


Transformation: TypeAlias = Rotation | Translation


def apply_transformations(
    points: NDArray[np.float_],
    transformations: list[Transformation] | None = None,
    *,
    inverse: bool = False,
) -> NDArray[np.float_]:
    """Applies a series of transformations to a sequence of vectors or a single vector.

    Args:
        points: point coordinate
        transformations: list of Transformation (Translation / Rotation)
        inverse (bool): Applies the inverse coordinate transform instead

    Returns:
        The collection of transformed points.
    """
    if transformations is None:
        transformations = []

    for transformation in transformations:
        points = transformation.apply(points, inverse=inverse)

    return points


def plot_plane_to_bz(
    cell: Sequence[Sequence[float]] | NDArray[np.float_],
    plane: str | list[NDArray[np.float_]],
    ax: Axes3D,
    special_points: dict[str, NDArray[np.float_]] | None = None,
    facecolor: ColorType = "red",
) -> None:
    """Plots a 2D cut plane onto a Brillouin zone.

    Args:
        cell: [TODO:description]
        plane: [TODO:description]
        ax: [TODO:description]
        special_points: [TODO:description]
        facecolor: [TODO:description]

    Returns:
        [TODO:description]
    """
    from ase.dft.bz import bz_vertices

    if isinstance(plane, str):
        plane_points: list[NDArray[np.float_]] = process_kpath(
            plane,
            np.array(cell),
            special_points=special_points,
        )[0]
    else:
        plane_points = plane

    d1, d2 = plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]

    faces = [p[0] for p in bz_vertices(np.linalg.inv(cell).T)]
    pts = polyhedron_intersect_plane(faces, np.cross(d1, d2), plane_points[0])

    collection = Poly3DCollection([pts])
    collection.set_facecolor(facecolor)
    ax.add_collection3d(collection, zs="z")


def plot_data_to_bz(
    data: xr.DataArray,
    cell: Sequence[Sequence[float]] | NDArray[np.float_],
    **kwargs: Incomplete,
) -> Path | tuple[Figure, Axes]:
    """A dimension agnostic tool used to plot ARPES data onto a Brillouin zone."""
    if len(data) == TWO_DIMENSION + 1:
        return plot_data_to_bz3d(data, cell, **kwargs)

    return plot_data_to_bz2d(data, cell, **kwargs)


def plot_data_to_bz2d(  # noqa: PLR0913
    data_array: xr.DataArray,
    cell: Sequence[Sequence[float]] | NDArray[np.float_],
    rotate: float | None = None,
    shift: NDArray[np.float_] | None = None,
    scale: float | None = None,
    ax: Axes | None = None,
    out: str | Path = "",
    bz_number: Sequence[float] | None = None,
    *,
    mask: bool = True,
    **kwargs: Incomplete,
) -> Path | tuple[Figure, Axes]:
    """Plots data onto a 2D Brillouin zone."""
    assert data_array.S.is_kspace, "You must k-space convert data before plotting to BZs"
    assert isinstance(data_array, xr.DataArray), "data_array must be xr.DataArray, not Dataset"

    if bz_number is None:
        bz_number = (0, 0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        bz2d_plot(cell, paths="all", ax=ax)
    assert isinstance(ax, Axes)

    if len(cell) == TWO_DIMENSION:
        cell = [[*list(c), 0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T

    # Prep coordinates and mask
    raveled = data_array.G.meshgrid(as_dataset=True)
    dims = data_array.dims
    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        rotation = np.array([(c, -s), (s, c)])

        raveled = raveled.G.transform_coords(dims, rotation)

    if scale is not None:
        raveled = raveled.G.scale_coords(dims, scale)

    if shift is not None:
        raveled = raveled.G.shift_coords(dims, shift)

    copied = data_array.values.copy()

    if mask:
        built_mask = apply_mask_to_coords(raveled, build_2dbz_poly(cell=cell), dims)
        copied[built_mask.T] = np.nan

    cmap = kwargs.get("cmap", matplotlib.colormaps["Blues"])
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps.get_cmap(cmap)

    cmap.set_bad((1, 1, 1, 0))

    delta_x = np.dot(np.array(bz_number), icell[:2, 0])
    delta_y = np.dot(np.array(bz_number), icell[:2, 1])

    ax.pcolormesh(
        raveled.data_vars[dims[0]].values + delta_x,
        raveled.data_vars[dims[1]].values + delta_y,
        copied.T,
        cmap=cmap,
    )

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


def plot_data_to_bz3d(
    data: xr.DataArray,
    cell: Sequence[Sequence[float]] | NDArray[np.float_],
    **kwargs: Incomplete,
) -> Path | tuple[Figure, Axes]:
    """Plots ARPES data onto a 3D Brillouin zone."""
    msg = "plot_data_to_bz3d is not implemented yet."
    logger.debug(f"id of data: {data.attrs.get('id', None)}")
    logger.debug(f"cell: {cell}")
    if kwargs:
        for k, v in kwargs.items():
            logger.debug(f"kwargs; k: {k}, v: {v}")
    raise NotImplementedError(msg)


def annotate_special_paths(
    ax: Axes | Axes3D,
    paths: list[str] | str = "",
    cell: NDArray[np.float_] | Sequence[Sequence[float]] | None = None,
    offset: dict[str, Sequence[float]] | None = None,
    special_points: dict[str, NDArray[np.float_]] | None = None,
    labels: Incomplete = None,
    **kwargs: Incomplete,
) -> None:
    """Annotates user indicated paths in k-space by plotting lines (or points) over the BZ.

    Args:
        ax: [TODO:description]
        paths: [TODO:description]
        cell: [TODO:description]
        offset: [TODO:description]
        special_points: [TODO:description]
        labels: [TODO:description]
        kwargs: [TODO:description]

    Raises:
        ValueError: [TODO:description]
    """
    if not paths:
        msg = "Must provide a proper path."
        raise ValueError(msg)

    if isinstance(paths, str):
        if labels is None:
            labels = paths

        converted_paths = process_kpath(paths, np.array(cell), special_points=special_points)
        logger.debug(f"converted_paths: {converted_paths}")

        if not isinstance(labels[0], list):
            labels = [labels]

        labels = [list(label) for label in labels]
        paths = list(zip(labels, converted_paths, strict=True))
        logger.debug(f"paths in annotate_special_paths {paths}")
    fontsize = kwargs.pop("fontsize", 14)

    if offset is None:
        offset = {}

    two_d = True
    if isinstance(ax, Axes3D):
        two_d = False

    assert isinstance(paths, list)
    for names, points in paths:
        x, y, z = np.array(points).T

        if two_d:
            ax.plot(x, y, c="r", ls="-", **kwargs)
        else:
            ax.plot(x, y, z, c="r", ls="-", **kwargs)

        for name, point in zip(names, points, strict=True):
            x, y, z = point
            display_name = name
            if display_name == "G":
                display_name = "\\Gamma"
            elif len(name) > 1:
                name = name[0] + "_" + name[1]

            off = offset.get(name, 0)
            try:
                if two_d:
                    x, y = x + off[0], y + off[1]
                else:
                    x, y, z = x + off[0], y + off[1], z + off[2]

            except TypeError:
                if two_d:
                    x, y = x + off, y + off
                else:
                    x, y, z = x + off, y + off, z + off

            if two_d:
                ax.text(
                    x,
                    y,
                    "$" + display_name + "$",
                    ha="center",
                    va="bottom",
                    color="r",
                    fontsize=fontsize,
                )
            else:
                ax.text(
                    x,
                    y,
                    z,
                    "$" + display_name + "$",
                    ha="center",
                    va="bottom",
                    color="r",
                    fontsize=fontsize,
                )


def bz2d_segments(
    cell: Sequence[Sequence[float]],
    transformations: list[Transformation] | None = None,
) -> tuple[list[NDArray[np.float_]], list[NDArray[np.float_]]]:
    """Calculates the line segments corresponding to a 2D BZ."""
    segments_x = []
    segments_y = []

    for points, _normal in twocell_to_bz1(np.array(cell))[0]:
        transformed_points = apply_transformations(points, transformations)
        x, y, z = np.concatenate([transformed_points, transformed_points[:1]]).T
        segments_x.append(x)
        segments_y.append(y)

    return segments_x, segments_y


def twocell_to_bz1(cell: NDArray[np.float_]) -> Incomplete:
    from ase.dft.bz import bz_vertices

    # 2d in x-y plane
    if len(cell) > TWO_DIMENSION:
        assert all(abs(cell[2][0:2]) < 1e-6)  # noqa: PLR2004
        assert all(abs(cell.T[2][0:2]) < 1e-6)  # noqa: PLR2004
    else:
        cell = np.array([[*list(c), 0] for c in cell] + [[0, 0, 1]])
    icell = np.linalg.inv(cell).T
    try:
        bz1 = bz_vertices(icell[:3, :3], dim=2)
    except TypeError:
        bz1 = bz_vertices(icell[:3, :3])
    return bz1, icell, cell
