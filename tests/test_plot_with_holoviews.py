"""Unit test for plotting/holoviews.py."""

import xarray as xr
from arpes.plotting import crosshair_view
from holoviews.core.layout import AdjointLayout


class TestCrosshairView:
    """Class for crosshair_view function."""

    def test_basic_crosshair_view(self, dataarray_cut2: xr.DataArray) -> None:
        img = crosshair_view(dataarray_cut2)
        assert isinstance(img, AdjointLayout)
