"""Image reading methods with different library fallbacks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = (
    "imread",
    "imread_to_xarray",
)


def imread(str_or_path: str | Path) -> NDArray[np.float64]:
    """A wrapper around `opencv.imread` and `imageio.imread`.

    As compared to those, this method

    1. Falls back to the first available option on the system
    2. Normalizes OpenCV images to RGB format
    3. Removes the alpha channel from imageio.imread data

    Args:
        str_or_path: pathlib.Path or str containing the image to be read

    Returns:
        The image as an np.ndarray.
    """
    try:
        import cv2

        using_cv2 = True
    except ImportError:
        try:
            import imageio

            using_cv2 = False
        except ImportError:
            warnings.warn(
                "You need OpenCV or imageio in order to read images in PyARPES.",
                stacklevel=2,
            )
            raise

    if using_cv2:
        arr = cv2.imread(str(str_or_path))
        return np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1)
    arr = imageio.imread(str(str_or_path))
    return arr[:, :, :3]


def imread_to_xarray(str_or_path: str | Path) -> xr.DataArray:
    """Like `imread`, except that this function wraps the result into a xr.DataArray instance.

    The read instance has x (pixel), y (pixel), and color (['R', 'G', 'B']) dimensions.

    Args:
        str_or_path : Path to file

    Returns:
        The data read to an `xr.DataArray` instance.
    """
    raw_arr = imread(str_or_path)
    sx, sy, _ = raw_arr.shape

    return xr.DataArray(
        raw_arr,
        coords={"x": np.asarray(range(sx)), "y": np.asarray(range(sy)), "color": ["r", "g", "b"]},
        dims=["x", "y", "color"],
    )
