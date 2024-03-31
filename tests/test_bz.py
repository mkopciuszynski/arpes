"""Test for utilitiy.bz."""

import numpy as np
from arpes.utilities import bz


def test_special_point_to_vector() -> None:
    """Test for special_point_to_vector.

    FCC bulk lattice is used for test.
    """
    icell = np.array([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
    special_point_k = bz.SpecialPoint("K", negate=False, bz_coord=(0.375, 0.375, 0.75))

    #    """Test for parse_single_path."""
    # def test_single_path() -> None:
    #    path = "GKM"
    #    assert bz.parse_single_path(path) == [[0, 0, 0], [1, 2, 3], [3, 4, 5]]
