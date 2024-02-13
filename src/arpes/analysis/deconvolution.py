"""Provides deconvolution implementations, especially for 2D Richardson-Lucy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.ndimage
import xarray as xr
from skimage.restoration import richardson_lucy

import arpes.xarray_extensions  # noqa: F401
from arpes.fits.fit_models.functional_forms import gaussian
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = (
    "deconvolve_ice",
    "deconvolve_rl",
    "make_psf1d",
)

TWO_DIMWENSION = 2


@update_provenance("Approximate Iterative Deconvolution")
def deconvolve_ice(
    data: xr.DataArray,
    psf: NDArray[np.float_],
    n_iterations: int = 5,
    deg: int | None = None,
) -> xr.DataArray | NDArray[np.float_]:
    """Deconvolves data by a given point spread function (PSF).

    The iterative convolution extrapolation method is used.
    The PSF is the impulse response of a focused optical imaging system.

    Args:
        data (DataType): input data
        psf(NDArray[np.float_): array as point spread function
        n_iterations: the number of convolutions to use for the fit
        deg: the degree of the fitting polynominial

    Returns:
        The deconvoled data in the same format.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    arr: NDArray[np.float_] = data.values
    if deg is None:
        deg = n_iterations - 3
    iteration_steps = list(range(1, n_iterations + 1))

    iteration_list = [arr]

    for _ in range(n_iterations - 1):
        iteration_list.append(scipy.ndimage.convolve(iteration_list[-1], psf))
    iteration_array = np.asarray(iteration_list)

    deconv = arr * 0
    for t, series in enumerate(iteration_array.T):
        coefs = np.polyfit(iteration_steps, series, deg=deg)
        poly = np.poly1d(coefs)
        deconv[t] = poly(0)

    if isinstance(data, np.ndarray):
        result = deconv
    else:
        result = data.copy(deep=True)
        result.values = deconv
    return result


@update_provenance("Lucy Richardson Deconvolution")
def deconvolve_rl(
    data: xr.DataArray,
    psf: xr.DataArray,
    n_iterations: int = 10,
) -> xr.DataArray:
    """Deconvolves data by a given point spread function using the Richardson-Lucy (RL) method.

    Args:
        data: input data
        psf:  The point spread function.
        n_iterations: the number of convolutions to use for the fit

    Returns:
        The Richardson-Lucy deconvolved data.
    """
    arr = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    data_image = arr.values
    psf_ = psf.values
    im_deconv = richardson_lucy(data_image, psf_, num_iter=n_iterations, filter_epsilon=None)
    return arr.S.with_values(im_deconv)


@update_provenance("Make 1D-Point Spread Function")
def make_psf1d(data: xr.DataArray, dim: str, sigma: float) -> xr.DataArray:
    """Produces a 1-dimensional gaussian point spread function for use in deconvolve_rl.

    Args:
        data (DataType): xarray object
        dim (str): dimension name
        sigma (float): sigma value

    Returns:
        A one dimensional point spread array.
    """
    arr = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    psf = arr.copy(deep=True) * 0 + 1
    other_dims = list(arr.dims)
    other_dims.remove(dim)
    for od in other_dims:
        psf = psf[{od: 0}]
    return psf * gaussian(psf.coords[dim], np.mean(psf.coords[dim]), sigma)


@update_provenance("Make Point Spread Function")
def make_psf(data: xr.DataArray, sigmas: dict[str, float]) -> xr.DataArray:
    """Produces an n-dimensional gaussian point spread function for use in deconvolve_rl.

    Not yet operational.

    Args:
        data (DataType): input data
        sigmas (dict[str, float]): sigma values for each dimension.

    Returns:
        The PSF to use.
    """
