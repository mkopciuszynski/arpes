"""Math snippets used elsewhere in PyARPES."""
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage.interpolation
import xarray as xr

from arpes.constants import K_BOLTZMANN_EV_KELVIN

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray


def derivative(f: Callable, arg_idx: int = 0) -> float:
    """Defines a simple midpoint derivative."""

    def d(*args: Incomplete):
        args = list(args)
        ref_arg = args[arg_idx]
        d = ref_arg / 100
        args[arg_idx] = ref_arg + d
        high = f(*args)
        args[arg_idx] = ref_arg - d
        low = f(*args)
        return (high - low) / (2 * d)

    return d


def polarization(up, down):
    """The equivalent normalized difference for a two component signal."""
    return (up - down) / (up + down)


def propagate_statistical_error(f):
    """To compute a function which propagates statistical error.

    It Uses numerical derivatives and sampling.
    """

    def compute_propagated_error(*args):
        running_sum = 0
        for i, arg in enumerate(args):
            df_darg_i = derivative(f, i)
            running_sum += df_darg_i(*args) ** 2 * arg

        return np.sqrt(running_sum)

    return compute_propagated_error


def shift_by(
    arr: NDArray[np.float_],
    value: xr.DataArray | NDArray[np.float_],
    axis: float = 0,
    by_axis=0,
    **kwargs: Incomplete,
) -> NDArray[np.float_]:
    """Shifts slices of `arr` perpendicular to `by_axis` by `value`.

    [TODO:description]

    Args:
        arr ([TODO:type]): [TODO:description]
        value ([TODO:type]): [TODO:description]
        axis ([TODO:type]): [TODO:description]
        by_axis ([TODO:type]): [TODO:description]
        **kwargs(): pass to ndimage.interpolation.shift
    """
    assert axis != by_axis
    arr_copy = arr.copy()
    if isinstance(value, xr.DataArray):
        value = value.values
    assert isinstance(value, np.ndarray)
    if not isinstance(value, Iterable):
        value = list(itertools.repeat(value, times=arr.shape[by_axis]))
    for axis_idx in range(arr.shape[by_axis]):
        slc = (slice(None),) * by_axis + (axis_idx,) + (slice(None),) * (arr.ndim - by_axis - 1)
        shift_amount = (0,) * axis + (value[axis_idx],) + (0,) * (arr.ndim - axis - 1)
        shift_amount = shift_amount[1:] if axis > by_axis else shift_amount[:-1]
        arr_copy[slc] = scipy.ndimage.interpolation.shift(arr[slc], shift_amount, **kwargs)
    return arr_copy


def inv_fermi_distribution(
    energy: NDArray[np.float_] | float,
    temperature: float,
    mu: float = 0.0,
) -> NDArray[np.float_]:
    """Expects energy in eV and temperature in Kelvin."""
    return np.exp((energy - mu) / (K_BOLTZMANN_EV_KELVIN * temperature)) + 1.0


def fermi_distribution(
    energy: NDArray[np.float_] | float,
    temperature: float,
) -> NDArray[np.float_] | float:
    """Expects energy in eV and temperature in Kelvin."""
    return 1.0 / inv_fermi_distribution(energy, temperature)
