"""Unit test for curve fitting."""

import numpy as np
import xarray as xr
from arpes.analysis import rebin
from arpes.fits import AffineBroadenedFD, broadcast_model

TOLERANCE = 2e-3


def test_broadcast_fitting(dataarray_cut: xr.DataArray) -> None:
    """Test broadcast fitting."""
    near_ef = dataarray_cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
    near_ef_rebin = rebin(near_ef, phi=5)

    fit_results = broadcast_model([AffineBroadenedFD], near_ef_rebin, "phi", progress=False)

    assert np.abs(fit_results.results.F.p("a_fd_center").mean().item() + 0.00508) < TOLERANCE

    fit_results = broadcast_model(
        [AffineBroadenedFD],
        near_ef_rebin,
        "phi",
        parallelize=True,
        progress=True,
    )
    assert fit_results.results.F.parameter_names == {
        "a_const_bkg",
        "a_conv_width",
        "a_fd_center",
        "a_fd_width",
        "a_lin_bkg",
        "a_offset",
    }
    assert fit_results.results.F.band_names == {"a_fd_"}
    assert fit_results.F.broadcast_dimensions == ["phi"]
    assert fit_results.F.fit_dimensions == ["eV"]
