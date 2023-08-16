"""Montage two ARPES data."""
import xarray as xr

__all__ = ("montage",)


def montage(
    arr_a: xr.DataArray,
    arr_b: xr.DataArray,
    occupation_ratio: float = 0.5,
) -> xr.DataArray:
    """Montage two arpes data.

    [TODO:description]

    Args:
        arr_a (xr.DataArray): [TODO:description]
        arr_b (xr.DataArray): [TODO:description]
        occupation_ratio(float): [TODO:description]

    Returns:
        [TODO:description]
    """
    assert 0 <= occupation_ratio <= 1, "occupation_ratio should be between 0 and 1"
