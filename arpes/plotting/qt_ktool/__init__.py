"""A live momentun conversion tool, useful for finding and setting offsets."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets

from arpes.plotting.bz import segments_standard
from arpes.utilities import group_by, normalize_to_spectrum
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.qt import SimpleApp, SimpleWindow, qt_info
from arpes.utilities.ui import CollectUI, horizontal, label, numeric_input, tabs, vertical

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from _typeshed import Incomplete
    from PySide6.QtWidgets import QLayout

    from arpes._typing import ANGLE, DataType

__all__ = (
    "KTool",
    "ktool",
)


qt_info.setup_pyqtgraph()


class KTool(SimpleApp):
    """Provides a live momentum converting tool.

    QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PySide6 for now we retain
    a number of the metaphors from BokehTool, including a "context" that stores the state, and can
    be used to programmatically interface with the tool.
    """

    TITLE = "KSpace-Tool"
    WINDOW_SIZE = (5, 6)
    WINDOW_CLS = SimpleWindow

    DEFAULT_COLORMAP = "viridis"

    def __init__(
        self,
        *,
        apply_offsets: bool = True,
        zone: str | Sequence[float] | None = None,
        **kwargs: Incomplete,
    ) -> None:
        """Set attributes to safe defaults and unwrap the Brillouin zone definition."""
        super().__init__()

        if isinstance(
            zone,
            tuple | list,
        ):
            self.segments_x, self.segments_y = zone
        elif isinstance(zone, str) and zone:
            self.segments_x, self.segments_y = segments_standard(zone)
        else:
            self.segments_x, self.segments_y = None, None

        self.conversion_kwargs = kwargs
        self.data = None
        self.content_layout: QLayout
        self.main_layout: QLayout
        self.apply_offsets = apply_offsets

    def configure_image_widgets(self) -> None:
        """We have two marginals because we deal with Fermi surfaces, they get configured here."""
        self.generate_marginal_for((), 0, 0, "xy", cursors=False, layout=self.content_layout)
        self.generate_marginal_for((), 1, 0, "kxy", cursors=False, layout=self.content_layout)

    def add_contextual_widgets(self) -> None:
        """The main UI layout for controls and tools."""
        convert_dims = ["theta", "beta", "phi", "psi"]
        if "eV" not in self.data.dims:
            convert_dims += ["chi"]
        if "hv" in self.data.dims:
            convert_dims += ["hv"]

        ui = {}
        with CollectUI(ui):
            controls = tabs(
                [
                    "Controls",
                    horizontal(
                        *[
                            vertical(
                                *[
                                    vertical(
                                        label(p),
                                        numeric_input(
                                            self.data.attrs.get(f"{p}_offset", 0.0),
                                            input_type=float,
                                            id=f"control-{p}",
                                        ),
                                    )
                                    for p in pair
                                ],
                            )
                            for pair in group_by(2, convert_dims)
                        ],
                    ),
                ],
            )

        def update_dimension_name(dim_name: str) -> Callable[[str | float], None]:
            def updater(value: str | float) -> None:
                self.update_offsets(dict([[dim_name, float(value)]]))

            return updater

        for dim in convert_dims:
            ui[f"control-{dim}"].subject.subscribe(update_dimension_name(dim))

        controls.setFixedHeight(qt_info.inches_to_px(1.75))

        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(controls, 1, 0)

    def update_offsets(self, offsets: dict[ANGLE, float]) -> None:
        """Pushes offsets to the display data and optionally, the original data."""
        self.data.S.apply_offsets(offsets)
        if self.apply_offsets:
            self.original_data.S.apply_offsets(offsets)
        self.update_data()

    def layout(self) -> QLayout:
        """Initialize the layout components."""
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        return self.main_layout

    def update_data(self) -> None:
        """The main redraw method for this tool.

        Converts data into momentum space and populates both the angle-space and momentum
        space views.

        If a Brillouin zone was requested, plots that over the data as well.
        """
        self.views["xy"].setImage(self.data)

        kdata = convert_to_kspace(self.data, **self.conversion_kwargs)
        if "eV" in kdata.dims:
            kdata = kdata.S.transpose_to_back("eV")

        self.views["kxy"].setImage(kdata.fillna(0))
        if self.segments_x is not None:
            bz_plot = self.views["kxy"].plot_item
            kx, ky = self.conversion_kwargs["kx"], self.conversion_kwargs["ky"]
            for segx, segy in zip(self.segments_x, self.segments_y, strict=True):
                bz_plot.plot((segx - kx[0]) / (kx[1] - kx[0]), (segy - ky[0]) / (ky[1] - ky[0]))

    def before_show(self) -> None:
        """Lifecycle hook for configuration before app show."""
        self.configure_image_widgets()
        self.add_contextual_widgets()
        if self.DEFAULT_COLORMAP is not None:
            self.set_colormap(self.DEFAULT_COLORMAP)

    def after_show(self) -> None:
        """Initialize application state after app show. Just redraw."""
        self.update_data()

    def set_data(self, data: DataType) -> None:
        """Sets the current data to a new value and resets binning.

        Above what happens in QtTool, we try to extract a Fermi surface, and
        repopulate the conversion.
        """
        original_data = normalize_to_spectrum(data)
        self.original_data = original_data

        if len(data.dims) > 2:  # noqa: PLR2004
            assert "eV" in original_data.dims
            data = data.sel(eV=slice(-0.05, 0.05)).sum("eV", keep_attrs=True)
            data.coords["eV"] = 0
        else:
            data = original_data

        if "eV" in data.dims:
            data = data.S.transpose_to_back("eV")

        self.data = data.copy(deep=True)

        if not self.conversion_kwargs:
            rng_mul = 1.0
            if data.coords["hv"] < 12:  # noqa: PLR2004
                rng_mul = 0.5
            if data.coords["hv"] < 7:  # noqa: PLR2004
                rng_mul = 0.25

            if "eV" in self.data.dims:
                self.conversion_kwargs = {
                    "kp": np.linspace(-2, 2, 400) * rng_mul,
                }
            else:
                self.conversion_kwargs = {
                    "kx": np.linspace(-2, 2, 300) * rng_mul,
                    "ky": np.linspace(-2, 2, 300) * rng_mul,
                }


def ktool(data: DataType, **kwargs: Incomplete) -> KTool:
    """Start the momentum conversion tool."""
    tool = KTool(**kwargs)
    tool.set_data(data)
    tool.start()
    return tool
