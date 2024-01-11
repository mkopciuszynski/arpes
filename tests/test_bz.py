"""Test for utilitiy.bz."""


import numpy as np

from arpes.utilities import bz


def test_as_3d() -> None:
    """Test for 'bz.as_3d'."""
    points_2d = np.array([[0, 1], [2, 3]])
    np.testing.assert_array_equal(bz.as_3d(points_2d), np.array([[0, 1, 0], [2, 3, 0]]))


def test_as_2d() -> None:
    """Test for ''bz.as_2d'."""
    points_3d = np.array([[0, 1, 2], [2, 3, 4]])
    np.testing.assert_array_equal(bz.as_2d(points_3d), np.array([[0, 1], [2, 3]]))


def test_orthorombic_cell() -> None:
    """Test for orthormbic_cell function."""
    np.testing.assert_array_equal(
        bz.orthorhombic_cell(5.4, 2.3, 3.0),
        np.array(
            [[5.4, 0, 0], [0, 2.3, 0], [0, 0, 3.0]],
        ),
    )


def test_hex_cell() -> None:
    """Test for hex cell function."""
    np.testing.assert_array_equal(
        bz.hex_cell(3.0, 4.0),
        np.array(
            [
                [3.0, 0.0, 0.0],
                [-1.5, 2.598076211353316, 0],
                [0, 0, 4.0],
            ],
        ),
    )

    np.testing.assert_array_equal(
        bz.hex_cell_2d(3.0),
        np.array(
            [
                [3.0, 0.0],
                [-1.5, 2.598076211353316],
            ],
        ),
    )

    # def test_single_path() -> None:
    #    """Test for parse_single_path."""
    #    path = "GKM"
    #    assert bz.parse_single_path(path) == [[0, 0, 0], [1, 2, 3], [3, 4, 5]]
