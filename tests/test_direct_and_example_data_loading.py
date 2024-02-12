"""Test for data loading."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from arpes.endstations.plugin.ALG_main import ALGMainChamber
from arpes.io import load_data, load_example_data

if TYPE_CHECKING:
    from collections.abc import Generator

    from .conftest import Sandbox


def test_load_data(
    sandbox_configuration: Generator[Sandbox, None, None],  # noqa: ARG001
) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]
    """
    test_data_location = (
        Path(__file__).parent / "resources" / "datasets" / "basic" / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)


def test_load_data_with_plugin_specified(
    sandbox_configuration: Generator[Sandbox, None, None],  # noqa: ARG001
) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]
    """
    test_data_location = (
        Path(__file__).parent / "resources" / "datasets" / "basic" / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")
    directly_specified_data = load_data(file=test_data_location, location=ALGMainChamber)

    assert isinstance(directly_specified_data, xr.Dataset)
    assert directly_specified_data.spectrum.shape == (240, 240)
    assert np.all(data.spectrum.values == directly_specified_data.spectrum.values)


def test_load_example_data(
    sandbox_configuration: Generator[Sandbox, None, None],  # noqa: ARG001
) -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]
    """
    data = load_example_data("cut")

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)
