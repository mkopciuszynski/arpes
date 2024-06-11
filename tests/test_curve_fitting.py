"""Unit test for curve fitting."""

import numpy as np
import xarray as xr
from arpes.analysis import rebin
from arpes.fits import AffineBroadenedFD, LorentzianModel, broadcast_model
from arpes.fits.utilities import parse_model

TOLERANCE = 2e-3


def test_parse_model() -> None:
    """Test parse_model."""
    assert parse_model(AffineBroadenedFD) == AffineBroadenedFD
    assert parse_model((AffineBroadenedFD, LorentzianModel)) == (AffineBroadenedFD, LorentzianModel)
    assert parse_model([AffineBroadenedFD, LorentzianModel]) == [AffineBroadenedFD, LorentzianModel]
    assert parse_model("AffineBroadenedFD + LorentzianModel") == [
        AffineBroadenedFD,
        "+",
        LorentzianModel,
    ]


def test_broadcast_fitting(dataarray_cut: xr.DataArray) -> None:
    """Test broadcast fitting."""
    near_ef = dataarray_cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
    near_ef_rebin = rebin(near_ef, phi=5)

    fit_results = broadcast_model([AffineBroadenedFD], near_ef_rebin, "phi", progress=False)
    a_band_data = fit_results.results.F.bands["a_"]
    np.testing.assert_almost_equal(
        a_band_data.center.values,
        np.array(
            [
                -0.00456954,
                -0.00303001,
                -0.00268052,
                -0.02043156,
                -0.00331799,
                -0.00397656,
                -0.00390413,
                0.00052012,
            ],
        ),
    )
    np.testing.assert_almost_equal(
        actual=a_band_data.sigma,
        desired=np.array((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)),
    )

    np.testing.assert_almost_equal(
        actual=a_band_data.amplitude,
        desired=np.array((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)),
    )
    assert np.abs(fit_results.results.F.p("a_center").mean().item() + 0.00508) < TOLERANCE

    fit_results = broadcast_model(
        [AffineBroadenedFD],
        near_ef_rebin,
        "phi",
        parallelize=True,
        progress=True,
    )
    assert fit_results.F.parameter_names == {
        "a_const_bkg",
        "a_conv_width",
        "a_center",
        "a_width",
        "a_lin_bkg",
        "a_offset",
    }
    assert fit_results.results.F.band_names == {"a_"}
    assert fit_results.F.broadcast_dimensions == ["phi"]
    assert fit_results.F.fit_dimensions == ["eV"]
    np.testing.assert_almost_equal(
        fit_results.F.mean_square_error().values,
        np.array(
            [
                1558314.89601851,
                1511866.71409967,
                1458591.79853167,
                457007.03427883,
                1279915.83565262,
                1319729.87019014,
                1229990.43899916,
                1193485.20821447,
            ],
        ),
    )
    np.testing.assert_allclose(
        fit_results.F.s("a_conv_width").values,
        np.array(
            [
                1.54832468e03,
                1.36921697e06,
                5.23737413e06,
                1.60255187e01,
                1.16205297e07,
                8.98171495e06,
                3.16863545e06,
                3.57829642e06,
            ],
        ),
    )
    params_ = fit_results.results.F.param_as_dataset("a_conv_width")
    np.testing.assert_almost_equal(
        params_["value"].values,
        np.array(
            [
                2.88798359,
                1.61713076,
                3.06574927,
                0.14156929,
                3.20488811,
                3.35551708,
                3.06293996,
                1.40944032,
            ],
        ),
    )
