"""This module contains utilities for generating momentum grids from angle-space data.

This process consists of:
    1. Determining the momentum axes which are necessary for a dataset based on which coordinate
       axes it has
    2. Determining the range over the output axes which is required for the data
    3. Determining an appropriate resolution or binning in the output grid
"""
from __future__ import annotations

import enum
import itertools

__all__ = [
    "is_dimension_convertible_to_mementum",
    "AxisType",
    "determine_axis_type",
    "determine_momentum_axes_from_measurement_axes",
]


def is_dimension_convertible_to_mementum(dimension_name: str) -> bool:
    """Determine whether a dimension can paticipate in the momentum conversion

    Originally, is_dimension_unconvertible(dimension_name: str) is defined.

    Args:
        dimension_name (str): [description]

    Returns:
        bool: True if the dimension name represents the angle (but not alpha) or hv
    """
    if dimension_name in ("phi", "theta", "beta", "chi", "psi", "hv"):
        return True
    return False


class AxisType(str, enum.Enum):
    """Models whether a given dimension is angle-like or momentum-like."""

    Angle = "angle"
    Momentum = "k"


def determine_axis_type(
    coordinate_names: list[str] | tuple[str, ...], permissive: bool = True
) -> AxisType:
    """Determines whether the input axes are better described as angle axes or momentum axes.

    Args:
        coordinate_names: The names of the coordinates
        permissive: Whether additional coordinates should be tossed out before checking

    Returns:
        What kind of axes they are.
    """
    coordinate_names = tuple(sorted(coordinate_names))
    mapping = {
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
    fixed_coordinate_names = tuple(t for t in coordinate_names if t in all_allowable)

    if fixed_coordinate_names != coordinate_names and not permissive:
        raise ValueError(
            f"""Received some coordinates {coordinate_names} which are
                not compatible with angle/k determination."""
        )
    return mapping[coordinate_names]


def determine_momentum_axes_from_measurement_axes(axis_names: list[str]) -> list[str]:
    """Associates the appropriate set of momentum dimensions given the angular dimensions."""
    return {
        ("phi",): ["kp"],
        ("beta", "phi"): ["kx", "ky"],
        ("phi", "theta"): ["kx", "ky"],
        ("phi", "psi"): ["kx", "ky"],
        ("hv", "phi"): ["kp", "kz"],
        ("beta", "hv", "phi"): ["kx", "ky", "kz"],
        ("hv", "phi", "theta"): ["kx", "ky", "kz"],
        ("hv", "phi", "psi"): ["kx", "ky", "kz"],
    }.get(tuple(sorted(axis_names)), [])
