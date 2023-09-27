"""Test for derivative procedure."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from arpes.analysis.derivative import dn_along_axis
from arpes.analysis.filters import gaussian_filter_arr

if TYPE_CHECKING:
    import xarray as xr
    from _typeshed import Incomplete


def test_dataarray_derivatives(sandbox_configuration: Incomplete) -> None:
    """Test for derivativation of xarray.

    Nick ran into an issue where he could not call dn_along_axis with a smooth function that
    expected a DataArray, but this is supposed to be supported. Issue was translation between
    np.ndarray and xr.DataArray internal to dn_along_axis.

    :param sandbox_configuration:
    :return:
    """

    def wrapped_filter(arr: xr.DataArray) -> xr.DataArray:
        return gaussian_filter_arr(arr, {"eV": 0.05, "phi": np.pi / 180})

    data = sandbox_configuration.load("basic/main_chamber_cut_0.fits").spectrum

    d2_data = dn_along_axis(data, "eV", wrapped_filter, order=2)

    # some random sample
    assert [pytest.approx(v, 1e-3) for v in (d2_data.values[50:55, 60:62].ravel())] == [
        46225.01571650838,
        -5.820766091346741e-11,
        3.2014213502407074e-09,
        46225.01571650361,
        46225.01571650687,
        -1.6298145055770874e-09,
        1.6298145055770874e-09,
        46225.0157165036,
        92450.03143301269,
        46225.01571650588,
    ]
