"""Unit test for statck_dispersion_plot."""

import numpy as np
import xarray as xr
from arpes.plotting import stack_plot


class TestStackDispersionPlot:
    """Test class for stack_dispersion_plot."""

    def test_statck_dispersion_plot_1(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for stack_dispersion_plot.

        Todo:
            Add more assert.
        """
        _, ax = stack_plot.stack_dispersion_plot(
            data=dataarray_cut2,
            max_stacks=20,
            scale_factor=20,
            title="2PPE Xe/Au(111)",
            linewidth=0.3,
            color="plasma",
            shift=0.0,
            mode="hide_line",
        )
        assert ax.get_title() == "2PPE Xe/Au(111)"

    def test_statck_dispersion_plot_2(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for stack_dispersion_plot.

        with fill_between_mode.

        Todo:
            Add more assert.
        """
        _, ax = stack_plot.stack_dispersion_plot(
            data=dataarray_cut2,
            max_stacks=10,
            linewidth=0.3,
            color="plasma",
            mode="fill_between",
            offset_correction="zero",
            shift=0.0,
        )
        assert len(ax.collections) == 10
        paths = ax.collections
        np.testing.assert_array_almost_equal(
            paths[0].get_paths()[0].vertices[:3],
            np.array([[9.0, 0.19602282], [9.0, 0.19632079], [9.002, 0.19634603]]),
        )
        np.testing.assert_array_almost_equal(
            actual=paths[-1].get_paths()[0].vertices[:3],
            desired=np.array([[9.0, -0.19602282], [9.0, -0.19578132], [9.002, -0.19573485]]),
        )
        xmin, xmax = ax.get_xlim()
        assert xmin == 8.95
        assert xmax == 10.05
        ymin, ymax = ax.get_ylim()
        assert ymin == -0.2157909190282624
        assert ymax == 0.21910736470983783


class TestFlatStackPlot:
    """Test class for flat_stack_plot."""

    def test_flat_stack_plot(self, dataarray_cut2: xr.DataArray) -> None:
        """Testing for flat_stack_plot.

        Todo:
            Add more assert.

        """
        _, ax = stack_plot.flat_stack_plot(
            dataarray_cut2,
            max_stacks=10,
            title="2PPE Xe/Au(111)",
            linewidth=0.5,
            color="plasma",
            label="label test",
            mode="line",
            figsize=(7, 5),
        )

        lines = ax.lines
        assert len(lines) == 10
