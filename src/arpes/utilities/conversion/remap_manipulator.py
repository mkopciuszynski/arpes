"""Contains utilities to determine equivalent coordinates between pairs of scans."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = ["remap_coords_to"]


def remap_coords_to(
    arr: DataType,
    reference_arr: DataType,
) -> dict[str, NDArray[np.float64] | xr.DataArray]:
    """Produces coordinates for the scan path of `arr` in the coordinate system of `reference_arr`.

    This needs to be thought out a bit more, namely to take into account better the
    manipulator scan degree of freeedom.

    Produces coords which provide the scan cut path for the array ``arr`` as seen in the coordinate
    system defined by the manipulator location in ``reference_arr``. This is useful for plotting
    locations of cuts in a FS.

    This code also assumes that a hemispherical analyzer was used, because it uses the coordinate
    'phi'.

    Args:
        arr: Scan that represents a cut we would like to understand
        reference_arr: Scan providing the desired destination coordinates

    Returns:
        Coordinates dict providing the path cut by the dataset ``arr``
    """
    irrelevant_coordinates = list({"hv", "eV"}.intersection(set(arr.dims)))
    arr = arr.sum(
        *irrelevant_coordinates,
        keep_attrs=True,
    )  # sum is not so fast, but ensures there is data

    assert arr.S.is_kspace == reference_arr.S.is_kspace

    full_coords = arr.S.full_coords
    full_reference_coords = reference_arr.S.full_coords

    def float_or_zero(value: str | float | None) -> float:
        if isinstance(value, float):
            return value
        return 0.0

    delta_chi = float_or_zero(full_coords["chi"]) - float_or_zero(full_reference_coords["chi"])
    delta_theta = float_or_zero(full_reference_coords["theta"]) - float_or_zero(
        full_coords["theta"],
    )

    if arr.S.is_kspace:
        raise NotImplementedError
    # rotation matrix is
    # cos -sin
    # sin  cos
    o_phi = full_coords["phi"] + delta_theta
    o_polar = full_coords["beta"]
    phi_coord = np.cos(-delta_chi) * o_phi - np.sin(-delta_chi) * o_polar
    polar_coord = np.sin(-delta_chi) * o_phi + np.cos(-delta_chi) * o_polar
    remapped_coords = deepcopy(full_reference_coords)
    remapped_coords.update({"phi": phi_coord.data, "beta": polar_coord.data})
    return remapped_coords
