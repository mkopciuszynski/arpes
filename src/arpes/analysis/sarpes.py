"""Contains very basic spin-ARPES analysis routines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_dataset
from arpes.utilities.math import polarization

if TYPE_CHECKING:
    from arpes._typing import DataType

__all__ = (
    "to_intensity_polarization",
    "to_up_down",
    "normalize_sarpes_photocurrent",
)


@update_provenance("Normalize SARPES by photocurrent")
def normalize_sarpes_photocurrent(data: DataType) -> DataType:
    """Normalizes the down channel so that it matches the up channel in terms of mean photocurrent.

    Destroys the integrity of "count" data because we have scaled individual arrivals.

    Args:
        data: The input data which does not need to consist of count data.

    Returns:
        Scaled data. Independently, photocurrent up and down channels are used to perform scaling.
    """
    copied = data.copy(deep=True)
    copied.down.values = (copied.down * (copied.photocurrent_up / copied.photocurrent_down)).values
    return copied


@update_provenance("Convert polarization data to up-down spin channels")
def to_up_down(data: DataType) -> xr.Dataset:
    """Converts from [intensity, polarization] representation to [up, down] representation.

    This is the inverse function to `to_intensity_polarization`, neglecting the role of the
    sherman function.

    Args:
        data: The input data

    Returns:
        The data after conversion to up-down representation.
    """
    assert "intensity" in data.data_vars
    assert "polarization" in data.data_vars

    return xr.Dataset(
        {
            "up": data.intensity * (1 + data.polarization),
            "down": data.intensity * (1 - data.polarization),
        },
    )


@update_provenance("Convert up-down spin channels to polarization")
def to_intensity_polarization(
    data: xr.Dataset,
    *,
    perform_sherman_correction: bool = False,
) -> xr.Dataset:
    """Converts from [up, down] representation to [intensity, polarization] representation.

    This is the inverse function to `to_up_down`.

    In this future, we should also make this also work with the timing signals.

    Args:
        data: The input data
        perform_sherman_correction(bool): if True, apply sherman correction (default to False)

    Returns:
        The data after conversion to intensity-polarization representation.
    """
    data_set = data if isinstance(data, xr.Dataset) else normalize_to_dataset(data)
    assert isinstance(data_set, xr.Dataset)

    assert "up" in data_set.data_vars
    assert "down" in data_set.data_vars

    intensity = data_set.up + data_set.down
    spectrum_polarization = polarization(data_set.up, data_set.down)

    sherman_correction = 1.0
    if perform_sherman_correction:
        sherman_correction = data_set.S.sherman_function

    return xr.Dataset(
        {"intensity": intensity, "polarization": spectrum_polarization / sherman_correction},
    )