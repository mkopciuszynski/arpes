"""Utilities to programmatically get access to an ARPES spectrum as an xr.DataArray."""

from __future__ import annotations

import inspect
import warnings
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from arpes._typing import XrTypes

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


__all__ = ("normalize_to_spectrum",)


def normalize_to_spectrum(data: XrTypes | str) -> xr.DataArray:
    """Tries to extract the actual ARPES spectrum from a dataset containing other variables."""
    from arpes.io import load_data

    logger.debug(f"inspect.stack(): {inspect.stack()}")
    if isinstance(data, str):
        msg = "You may use a file for the data as a argument of  "
        msg += f"{inspect.stack()[1].function} in {inspect.stack()[1].filename}\n"
        msg += "Remember to use a DataArray not a Dataset nor the filename, "
        msg += "attempting to extract spectrum and copy attributes.\n"
        msg += "Not so sure if this is what you have really expected."
        return normalize_to_spectrum(load_data(data))

    msg = "You use Dataset as a argument of "
    msg += f"{inspect.stack()[1].function} in {inspect.stack()[1].filename}\n"
    msg += "Remember to use a DataArray not a Dataset, "
    msg += "attempting to extract spectrum and copy attributes.\n"
    warnings.warn(
        msg,
        stacklevel=2,
    )

    if isinstance(data, xr.Dataset):
        if "up" in data.data_vars:
            assert isinstance(data.up, xr.DataArray)
            return data.up
        return data.S.spectrum
    assert isinstance(data, xr.DataArray)
    return data
