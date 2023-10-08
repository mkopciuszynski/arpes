"""Provides xarray aware pyqtgraph plotting widgets."""
from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING

# pylint: disable=import-error
import numpy as np
import pyqtgraph as pg
from scipy import interpolate

from .utils import PlotOrientation

if TYPE_CHECKING:
    from _typeshed import Incomplete

    from arpes._typing import DataType

__all__ = (
    "DataArrayImageView",
    "DataArrayPlot",
)


class CoordAxis(pg.AxisItem):
    def __init__(self, dim_index: int, *args: Incomplete, **kwargs: Incomplete) -> None:
        self.dim_index = dim_index
        self.coord = None
        self.interp = None
        super().__init__(*args, **kwargs)

    def setImage(self, image: DataType) -> None:
        self.coord = image.coords[image.dims[self.dim_index]].values
        assert isinstance(self.coord, Sized)
        self.interp = interpolate.interp1d(
            np.arange(0, len(self.coord)),
            self.coord,
            fill_value="extrapolate",
        )

    def tickStrings(self, values, scale, spacing) -> list[str]:
        try:
            return [f"{f:.3f}" for f in self.interp(values)]
        except TypeError:
            return super().tickStrings(values, scale, spacing)


class DataArrayPlot(pg.PlotWidget):
    """A plot for 1D xr.DataArray instances with a coordinate aware axis."""

    def __init__(
        self,
        orientation: str,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> None:
        """Use custom axes so that we can provide coordinate-ful rather than pixel based values."""
        self.orientation = orientation

        axis_or = "bottom" if orientation == PlotOrientation.Horizontal else "left"
        self._coord_axis = CoordAxis(dim_index=0, orientation=axis_or)

        super().__init__(axisItems=dict([[axis_or, self._coord_axis]]), *args, **kwargs)

    def plot(
        self,
        data: DataType,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> pg.PlotDataItem:
        """Updates the UI with new data.

        Data also needs to be forwarded to the coordinate axis in case of transpose
        or changed range of data.
        """
        y = data.values
        self._coord_axis.setImage(data)

        if self.orientation == PlotOrientation.Horizontal:
            return self.plotItem.plot(
                np.arange(0, len(y)),
                y,
                *args,
                pen=pg.mkPen(color=(68, 1, 84), width=3),
                **kwargs,
            )
        return self.plotItem.plot(
            y,
            np.arange(0, len(y)),
            *args,
            pen=pg.mkPen(color=(68, 1, 84), width=3),
            **kwargs,
        )


class DataArrayImageView(pg.ImageView):
    """ImageView that transparently handles xarray data.

    It includes setting axis and coordinate information.
    This makes it easier to build interactive applications around realistic scientific datasets.
    """

    def __init__(
        self,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> None:
        """Use custom axes so that we can provide coordinate-ful rather than pixel based values."""
        self._coord_axes = {
            "left": CoordAxis(dim_index=1, orientation="left"),
            "bottom": CoordAxis(dim_index=0, orientation="bottom"),
        }
        plot_item = pg.PlotItem(axisItems=self._coord_axes)
        self.plot_item = plot_item
        super().__init__(view=plot_item, *args, **kwargs)

        self.view.invertY(b=False)

    def setImage(
        self,
        img: DataType,
        *args: Incomplete,
        keep_levels: bool = False,
        **kwargs: Incomplete,
    ) -> None:
        """Accepts an xarray.DataArray instead of a numpy array."""
        if keep_levels:
            levels = self.getLevels()

        for axis in self._coord_axes.values():
            axis.setImage(img)

        super().setImage(img.values, *args, **kwargs)

        if keep_levels:
            self.setLevels(*levels)

    def recompute(self) -> None:
        """A hook to recompute UI state, not used by this widget."""
