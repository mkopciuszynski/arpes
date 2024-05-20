"""This module contains utilities for generating momentum grids from angle-space data.

This process consists of:
    1. Determining the momentum axes which are necessary for a dataset based on which coordinate
       axes it has
    2. Determining the range over the output axes which is required for the data
    3. Determining an appropriate resolution or binning in the output grid
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Hashable

    from _collections_abc import dict_keys

__all__ = [
    "determine_axis_type",
    "determine_momentum_axes_from_measurement_axes",
    "is_dimension_convertible_to_momentum",
]


def is_dimension_convertible_to_momentum(dimension_name: str) -> bool:
    """Determine whether a dimension can paticipate in the momentum conversion.

    Originally, is_dimension_unconvertible(dimension_name: str) is defined.
        {"phi", "theta", "beta", "chi", "psi", "hv"} can be converted to momemtum.

    Args:
        dimension_name (str): [description]

    Returns:
        bool: True if the dimension name represents the angle (but not alpha) or hv
    """
    return dimension_name in {"phi", "theta", "beta", "chi", "psi", "hv"}


AxisType = Literal["angle", "k"]


def determine_axis_type(
    coordinate_names: dict_keys[Hashable, set[float]] | list[str],
    *,
    permissive: bool = True,
) -> AxisType:
    """Determines whether the input axes are better described as angle axes or momentum axes.

    Args:
        coordinate_names: The names of the coordinates
        permissive: Whether additional coordinates should be tossed out before checking

    Returns:
        What kind of axes they are.
    """
    coordinates = tuple(sorted(str(x) for x in coordinate_names))
    mapping: dict[tuple[str, ...], AxisType] = {
        ("beta", "phi"): "angle",
        ("chi", "phi"): "angle",
        ("phi", "psi"): "angle",
        ("phi", "theta"): "angle",
        ("kx", "ky"): "k",  # <=  should be "kp" ?
        ("kx", "kz"): "k",
        ("ky", "kz"): "k",
        ("kx", "ky", "kz"): "k",
    }

    all_allowable = set(itertools.chain(*mapping.keys()))
    fixed_coordinate_names: tuple[Hashable, ...] = tuple(
        t for t in coordinates if t in all_allowable
    )

    if fixed_coordinate_names != coordinates and not permissive:
        msg = f"Received some coordinates {coordinates} which are"
        msg += "not compatible with angle/k determination."
        raise ValueError(
            msg,
        )
    return mapping[coordinates]


def determine_momentum_axes_from_measurement_axes(
    axis_names: list[Literal["phi", "beta", "psi", "theta", "hv"]],
) -> list[Literal["kp", "kx", "ky", "kz"]]:
    """Associates the appropriate set of momentum dimensions given the angular dimensions."""
    sorted_axis_names = tuple(sorted(axis_names))
    phi_k_dict: dict[tuple[str, ...], list[Literal["kp", "kx", "ky", "kz"]]] = {
        ("phi",): ["kp"],
        ("beta", "phi"): ["kx", "ky"],
        ("phi", "theta"): ["kx", "ky"],
        ("phi", "psi"): ["kx", "ky"],
        ("hv", "phi"): ["kp", "kz"],
        ("beta", "hv", "phi"): ["kx", "ky", "kz"],
        ("hv", "phi", "theta"): ["kx", "ky", "kz"],
        ("hv", "phi", "psi"): ["kx", "ky", "kz"],
    }
    if sorted_axis_names in phi_k_dict:
        return phi_k_dict[sorted_axis_names]
    return []
