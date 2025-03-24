"""Automated utilities for calculating Fermi edge corrections."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from lmfit.models import LinearModel, QuadraticModel
from matplotlib.axes import Axes

from arpes.analysis import fit_fermi_edge
from arpes.constants import TWO_DIMENSION
from arpes.correction.intensity_map import shift_by
from arpes.fits import GStepBModel, broadcast_model
from arpes.provenance import Provenance, provenance, update_provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    import lmfit as lf
    from _typeshed import Incomplete

T = TypeVar("T")


def _exclude_from_set(
    excluded: set[str],
) -> Callable[[Iterable[str | Hashable]], list[str | Hashable]]:
    def exclude(_: Iterable[str | Hashable]) -> list[str | Hashable]:
        return list(set(_).difference(excluded))

    return exclude


exclude_hemisphere_axes = _exclude_from_set({"phi", "eV"})
exclude_hv_axes = _exclude_from_set({"hv", "eV"})


__all__ = (
    "apply_direct_fermi_edge_correction",
    "apply_photon_energy_fermi_edge_correction",
    "apply_quadratic_fermi_edge_correction",
    "build_direct_fermi_edge_correction",
    "build_photon_energy_fermi_edge_correction",
    "build_quadratic_fermi_edge_correction",
    "find_e_fermi_linear_dos",
)


def find_e_fermi_linear_dos(
    edc: xr.DataArray,
    guess: float | None,
    ax: Axes | None = None,
    *,
    plot: bool = False,
) -> float:
    """Estimate the Fermi level under the assumption of a linear density of states.

    Does a reasonable job of finding E_Fermi in-situ for graphene/graphite or other materials with
    a linear DOS near the chemical potential. You can provide an initial guess via guess, or one
    will be chosen half way through the EDC.

    The Fermi level is estimated as the location where the DoS crosses below an estimated background
    level

    Args:
        edc: Input data
        guess: Approximate location
        ax: matplotlib Axes object.
        plot: Whether to plot the fit, useful for debugging.

    Returns:
        The Fermi edge position.
    """
    if guess is None:
        guess = edc.eV.values[len(edc.eV) // 2]

    edc = edc - np.percentile(edc.values, (20,))[0]
    # Note that xr.Dataset.values is method not instance.
    mask = edc > np.percentile(edc.sel(eV=slice(None, guess)), 20)
    mod = LinearModel().guess_fit(edc[mask])

    chemical_potential = -mod.params["intercept"].value / mod.params["slope"].value

    if plot:
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        edc.S.plot(ax=ax)
        ax.axvline(chemical_potential, linestyle="--", color="red")
        ax.axvline(guess, linestyle="--", color="gray")

    return chemical_potential


def apply_direct_fermi_edge_correction(
    arr: xr.DataArray,
    correction: xr.DataArray | None = None,
    *args: Incomplete,
    **kwargs: Incomplete,
) -> xr.DataArray:
    """Applies a direct fermi edge correction stencil."""
    correction = correction or build_direct_fermi_edge_correction(
        arr,
        *args,
        **kwargs,
    )

    assert isinstance(correction, xr.Dataset)
    shift_amount = -correction / arr.G.stride(generic_dim_names=False)["eV"]  # pylint: disable=invalid-unary-operand-type
    energy_axis_index = list(arr.dims).index("eV")

    correction_axis = list(arr.dims).index(correction.dims[0])

    corrected_arr = xr.DataArray(
        data=shift_by(
            arr=arr.values,
            value=shift_amount,
            axis=energy_axis_index,
            by_axis=correction_axis,
            order=1,
        ),
        coords=arr.coords,
        dims=arr.dims,
        attrs=arr.attrs,
    )

    if "id" in corrected_arr.attrs:
        del corrected_arr.attrs["id"]
    provenance_context: Provenance = {
        "what": "Shifted Fermi edge to align at 0 along hv axis",
        "by": "apply_photon_energy_fermi_edge_correction",
        "correction": correction,  # TODO: NEED check
    }

    provenance(corrected_arr, arr, provenance_context)

    return corrected_arr


@update_provenance("Build direct Fermi edge correction")
def build_direct_fermi_edge_correction(
    arr: xr.DataArray,
    energy_range: slice | None = None,
    along: str = "phi",
    *,
    plot: bool = False,
) -> xr.DataArray:
    """Builds a direct fermi edge correction stencil.

    This means that fits are performed at each value of the 'phi' coordinate
    to get a list of fits. Bad fits are thrown out to form a stencil.

    This can be used to shift coordinates by the nearest value in the stencil.

    Args:
        arr (xr.DataArray) : input DataArray
        energy_range (slice): Energy range, which is used in xr.DataArray.sel(). default (-0.1, 0.1)
        plot (bool): if True, show the plot
        along (str): axis for non energy axis

    Returns:
        The array of fitted edge coordinates.
    """
    assert len(arr.dims) == TWO_DIMENSION, "arr should be 2D."
    edge_fit = fit_fermi_edge(arr, energy_range=energy_range).modelfit_result

    def sieve(_: Incomplete, v: Incomplete) -> bool:
        return v.item().params["center"].stderr < 0.001  # noqa: PLR2004

    corrections: xr.DataArray = edge_fit.G.filter_coord(along, sieve).G.map(
        lambda x: x.params["center"].value,
    )

    if plot:
        corrections.S.plot()

    return corrections


def build_quadratic_fermi_edge_correction(
    arr: xr.DataArray,
    fit_limit: float = 0.001,
    eV_slice: slice | None = None,  # noqa: N803
    *,
    plot: bool = False,
) -> lf.model.ModelResult:
    """Calculates a quadratic Fermi edge correction.

    Edge fitting and then quadratic fitting of edges.
    """
    # TODO: improve robustness here by allowing passing in the location of the fermi edge guess
    # We could also do this automatically by using the same method we use for step detection to find
    # the edge of the spectrometer image

    if eV_slice is None:
        approximate_fermi_level = arr.S.find_spectrum_energy_edges().max()
        eV_slice = slice(approximate_fermi_level - 0.4, approximate_fermi_level + 0.4)
    else:
        approximate_fermi_level = 0
    sum_axes = exclude_hemisphere_axes(arr.dims)
    edge_fit = broadcast_model(
        model_cls=GStepBModel,
        data=arr.sum(sum_axes).sel(eV=eV_slice),
        broadcast_dims="phi",
        params={"center": {"value": approximate_fermi_level}},
    )

    size_phi = len(arr.coords["phi"])
    not_nanny = (np.logical_not(np.isnan(arr)) * 1).sum("eV") > size_phi * 0.30
    condition = np.logical_and(edge_fit.F.s("center") < fit_limit, not_nanny)

    quadratic_corr = QuadraticModel().guess_fit(edge_fit.F.p("center"), weights=condition * 1)
    if plot:
        edge_fit.F.p("center").plot()
        plt.plot(arr.coords["phi"], quadratic_corr.best_fit)

    return quadratic_corr


@update_provenance("Build photon energy Fermi edge correction")
def build_photon_energy_fermi_edge_correction(
    arr: xr.DataArray,
    energy_window: float = 0.2,
) -> xr.Dataset:
    """Builds Fermi edge corrections across photon energy.

    (corrects monochromator miscalibration)
    """
    return broadcast_model(
        model_cls=GStepBModel,
        data=arr.sum(exclude_hv_axes(arr.dims)).sel(eV=slice(-energy_window, energy_window)),
        broadcast_dims="hv",
    )


def apply_photon_energy_fermi_edge_correction(
    arr: xr.DataArray,
    correction: xr.Dataset | None = None,
    **kwargs: Incomplete,
) -> xr.DataArray:
    """Applies Fermi edge corrections across photon energy_window.

    (corrects monochromator miscalibration)
    """
    if correction is None:
        correction = build_photon_energy_fermi_edge_correction(arr, **kwargs)
    assert isinstance(correction, xr.Dataset)
    correction_values = correction.G.map(lambda x: x.params["center"].value)
    if "corrections" not in arr.attrs:
        arr.attrs["corrections"] = {}

    arr.attrs["corrections"]["hv_correction"] = list(correction_values.values)

    shift_amount = -correction_values / arr.G.stride(generic_dim_names=False)["eV"]
    energy_axis_index = arr.dims.index("eV")
    hv_axis_index = arr.dims.index("hv")

    corrected_arr = xr.DataArray(
        data=shift_by(
            arr=arr.values,
            value=shift_amount,
            axis=energy_axis_index,
            by_axis=hv_axis_index,
            order=1,
        ),
        coords=arr.coords,
        dims=arr.dims,
        attrs=arr.attrs,
    )

    if "id" in corrected_arr.attrs:
        del corrected_arr.attrs["id"]
    provenance_context: Provenance = {
        "what": "Shifted Fermi edge to align at 0 along hv axis",
        "by": "apply_photon_energy_fermi_edge_correction",
        "correction": list(correction_values.values),
    }

    provenance(corrected_arr, arr, provenance_context)

    return corrected_arr


def apply_quadratic_fermi_edge_correction(
    arr: xr.DataArray,
    correction: lf.model.ModelResult | None = None,
    offset: float | None = None,
) -> xr.DataArray:
    """Applies a Fermi edge correction using a quadratic fit for the edge."""
    assert isinstance(arr, xr.DataArray)
    if correction is None:
        correction = build_quadratic_fermi_edge_correction(arr)

    if "corrections" not in arr.attrs:
        arr.attrs["corrections"] = {}

    arr.attrs["corrections"]["FE_Corr"] = correction.best_values

    delta_E: float = arr.coords["eV"].values[1] - arr.coords["eV"].values[0]
    dims = list(arr.dims)
    energy_axis_index = dims.index("eV")
    phi_axis_index = dims.index("phi")

    shift_amount_E = correction.eval(x=arr.coords["phi"].values)

    if offset is not None:
        shift_amount_E = shift_amount_E - offset

    shift_amount = -shift_amount_E / delta_E

    corrected_arr = xr.DataArray(
        shift_by(
            arr.values,
            shift_amount,
            axis=energy_axis_index,
            by_axis=phi_axis_index,
            order=1,
        ),
        arr.coords,
        arr.dims,
        attrs=arr.attrs,
    )

    if "id" in corrected_arr.attrs:
        del corrected_arr.attrs["id"]
    provenance_context: Provenance = {
        "what": "Shifted Fermi edge to align at 0",
        "by": "apply_quadratic_fermi_edge_correction",
        "correction": correction.best_values,
    }

    provenance(corrected_arr, arr, provenance_context)

    return corrected_arr
