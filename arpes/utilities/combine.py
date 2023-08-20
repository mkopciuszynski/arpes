"""Syntax suger for combination of ARPES data."""
import xarray as xr

__all__ = ("concat_along_phi",)


def concat_along_phi(
    arr_a: xr.DataArray,
    arr_b: xr.DataArray,
    occupation_ratio: float | None = None,
) -> xr.DataArray:
    """Montage two arpes data.

    [TODO:description]

    Args:
        arr_a (xr.DataArray): [TODO:description]
        arr_b (xr.DataArray): [TODO:description]
        occupation_ratio(float | None): [TODO:description]

    Returns:
        Concatenated ARPES array
    """
    if occupation_ratio is not None:
        assert 0 <= occupation_ratio <= 1, "occupation_ratio should be between 0 and 1 (or None)."
    id_arr_a = arr_a.attrs["id"]
    id_arr_b = arr_b.attrs["id"]
    id_add = _combine_id(id_arr_a, id_arr_b)
    if occupation_ratio is None:
        concat_array = xr.concat(
            [arr_a, arr_b],
            dim="phi",
            coords="minimal",
            combine_attrs="drop_conflicts",
        ).sortedby("phi")

    concat_array.attrs["id"] = id_add
    return concat_array


def _combine_id(id_a: int | tuple[int, ...], id_b: int | tuple[int]) -> tuple[int, ...]:
    if isinstance(id_a, int) and isinstance(id_b, int):
        return tuple(sorted([id_a, id_b]))
    if isinstance(id_a, tuple) and isinstance(id_b, int):
        return tuple(sorted([*list(id_a), id_b]))
    if isinstance(id_b, tuple) and isinstance(id_a, int):
        return tuple(sorted([*list(id_b), id_a]))
    if isinstance(id_a, tuple) and isinstance(id_b, tuple):
        return tuple(sorted(list(id_b) + list(id_a)))
    msg = "id_a and id_b must be int or tuple"
    raise RuntimeError(msg)
