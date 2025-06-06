"""Data prep routines for hemisphere data."""

from __future__ import annotations

from itertools import pairwise

import xarray as xr

from arpes.provenance import update_provenance

__all__ = ["stitch_maps"]


@update_provenance("Stitch maps together")
def stitch_maps(
    arr: xr.DataArray,
    arr2: xr.DataArray,
    dimension: str = "beta",
) -> xr.DataArray:
    """Stitches together two maps by appending and potentially dropping frames in the first dataset.

    This is useful for beamline work when the beam is lost or in L-ARPES if laser output is blocked
    for part of a scan and a subsequent scan was taken to repair the problem.

    Args:
        arr: Incomplete map
        arr2: completion of first map
        dimension(str): dimension for alignment

    Returns:
        xr.DataArray: The stitched map.
    """
    # as a first step we need to align the coords of map2 onto those of map1
    coord1 = arr.coords[dimension].data.copy()
    coord2 = arr2.coords[dimension].data.copy()

    first_repair_coordinate = coord2.data[0]
    i, lower, higher = None, None, None

    # search for the breakpoint
    for i, (lower, higher) in enumerate(pairwise(coord1)):
        if higher > first_repair_coordinate:
            break
        assert isinstance(i, int)
        delta_low, delta_high = lower - first_repair_coordinate, higher - first_repair_coordinate
    if abs(delta_low) < abs(delta_high):
        delta = delta_low
    else:
        delta = delta_high
        i += 1

    shifted_repair_map = arr2.copy()
    shifted_repair_map.coords[dimension].data += delta

    attrs = arr.attrs.copy()
    good_data_slice = {}
    good_data_slice[dimension] = slice(None, i)

    selected = arr.isel(good_data_slice)
    selected.attrs.clear()
    shifted_repair_map.attrs.clear()
    concatted = xr.concat([selected, shifted_repair_map], dim=dimension)
    return xr.DataArray(
        concatted.data,
        concatted.coords,
        concatted.dims,
        attrs=attrs,
    )
