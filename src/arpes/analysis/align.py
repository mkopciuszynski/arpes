"""This module contains methods that get unitful alignments of one array against another.

This is very useful for determining spectral shifts before doing serious curve fitting analysis or
similar.

Implementations are included for each of 1D and 2D arrays, but this could be simply extended to ND
if we need to. I doubt that this is necessary and don't mind the copied code too much at the
present.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from lmfit.models import QuadraticModel
from scipy import signal

from arpes.constants import TWO_DIMENSION

__all__ = ("align",)


def align2d(a: xr.DataArray, b: xr.DataArray, *, subpixel: bool = True) -> tuple[float, float]:
    """Returns the unitful offset of b in a for 2D arrays using 2D correlation.

    Args:
        a: The first input array.
        b: The second input array.
        subpixel(bool): If True, will perform subpixel alignment using a curve fit to the peak.

    Returns:
        The offset of a 2D array against another.
    """
    corr = signal.correlate2d(
        a.values - np.mean(a.values),
        b.values - np.mean(b.values),
        boundary="fill",
        mode="same",
    )

    y, x = np.unravel_index(np.argmax(corr), corr.shape)

    if subpixel:
        marg = xr.DataArray(
            corr[y - 10 : y + 10, x],
            coords={"index": np.linspace(-10, 9, 20)},
            dims=["index"],
        )
        marg = marg / np.max(marg)
        mod = QuadraticModel().guess_fit(marg)
        true_y = y + -mod.params["b"].value / (2 * mod.params["a"].value)

        marg = xr.DataArray(
            corr[y, x - 10 : x + 10],
            coords={"index": np.linspace(-10, 9, 20)},
            dims=["index"],
        )
        marg = marg / np.max(marg)
        mod = QuadraticModel().guess_fit(marg)
        true_x = x + -mod.params["b"].value / (2 * mod.params["a"].value)

        y, x = true_y, true_x

    return (
        (float(y) - a.values.shape[0] / 2.0) * a.G.stride(generic_dim_names=False)[a.dims[0]],
        (float(x) - a.values.shape[1] / 2.0) * a.G.stride(generic_dim_names=False)[a.dims[1]],
    )


def align1d(a: xr.DataArray, b: xr.DataArray, *, subpixel: bool = True) -> float:
    """Returns the unitful offset of b in a for 1D arrays using 1D correlation.

    Args:
        a: The first input array.
        b: The second input array.
        subpixel: If True, will perform subpixel alignment using a curve fit to the peak.

    Returns:
        The offset of an array against another.
    """
    corr = np.correlate(a.values - np.mean(a.values), b.values - np.mean(b.values), mode="same")
    (x,) = np.unravel_index(np.argmax(corr), corr.shape)

    if subpixel:
        marg = xr.DataArray(
            corr[x - 10 : x + 10],
            coords={"index": np.linspace(-10, 9, 20)},
            dims=["index"],
        )
        marg = marg / np.max(marg)
        mod = QuadraticModel().guess_fit(marg)
        x += -mod.params["b"].value / (2 * mod.params["a"].value)

    return (float(x) - a.values.shape[0] / 2.0) * a.G.stride(generic_dim_names=False)[a.dims[0]]


def align(a: xr.DataArray, b: xr.DataArray, **kwargs: bool) -> tuple[float, float] | float:
    """Returns the unitful offset of b in a for ndarrays.

    Args:
        a(xr.DataArray): The first input array.
        b(xr.DataArray): The second input array.
        kwargs(bool): Pass to align2d (and currently only subpixel = True/False is accepted)

    Returns:
        The offset of an array against another.
    """
    if len(a.dims) == 1:
        return align1d(a, b, **kwargs)

    assert len(a.dims) == TWO_DIMENSION
    return align2d(a, b, **kwargs)
