"""Unit test for xarray_extensions.py."""
import numpy as np
import pytest
import xarray as xr

from arpes.io import example_data


@pytest.fixture()
def dataset_cut() -> xr.Dataset:
    """A fixture for loading Dataset."""
    return example_data.cut


@pytest.fixture()
def dataarray_cut() -> xr.DataArray:
    """A fixture for loading DataArray."""
    return example_data.cut.spectrum


class TestAngleCorrection:
    """Test class for angle correction."""

    def test_correct_angle_by_phi_offset(self, dataarray_cut: xr.DataArray) -> None:
        """Test for correct_angle_by(phi_offset)."""
        dataarray_cut.S.correct_angle_by("phi_offset")
        np.testing.assert_almost_equal(
            dataarray_cut.coords["phi"][0:3].values,
            np.array([-0.18334318, -0.18159786, -0.17985253]),
        )

    def test_correct_angle_by_chi_offset(self, dataarray_cut: xr.DataArray) -> None:
        """Test for correct_angle_by("chi_offset)."""
        assert dataarray_cut.attrs["chi"] != 0
        dataarray_cut.S.correct_angle_by("chi_offset")
        assert dataarray_cut.attrs["chi"] == 0
        assert dataarray_cut.coords["chi"] == 0

    def test_correct_angle_by_theta(self, dataarray_cut: xr.DataArray) -> None:
        """Test for correct_angle_by(theta)."""
        dataarray_cut.attrs["theta"] = dataarray_cut.attrs["phi_offset"]
        dataarray_cut.S.correct_angle_by("theta")
        np.testing.assert_almost_equal(
            dataarray_cut.coords["phi"][0:3].values,
            np.array([-0.18334318, -0.18159786, -0.17985253]),
        )
        assert dataarray_cut.attrs["theta"] == 0

    def test_correct_angle_by_beta(self, dataarray_cut: xr.DataArray) -> None:
        """Test for correct_angle_by(beta)."""
        dataarray_cut.attrs["beta"] = dataarray_cut.attrs["phi_offset"]
        dataarray_cut.coords["alpha"] = dataarray_cut.attrs["alpha"] = np.pi / 2
        assert dataarray_cut.S.is_slit_vertical
        dataarray_cut.S.correct_angle_by("beta")
        np.testing.assert_almost_equal(
            dataarray_cut.coords["phi"][0:3].values,
            np.array([-0.18334318, -0.18159786, -0.17985253]),
        )
        assert dataarray_cut.attrs["beta"] == 0


class TestAngleUnitforDataArray:
    """Test class for angle_unit for DataArray."""

    def test_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle unit property for DataArray."""
        assert dataarray_cut.S.angle_unit == "Radians"

    def test_setter_of_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle_unit setter."""
        dataarray_cut.S.angle_unit = "Degrees"
        assert dataarray_cut.S.angle_unit == "Degrees"
        assert dataarray_cut.attrs["angle_unit"] == "Degrees"

    def test_swap_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for swap_angle_unit (DataArray version)."""
        original_phi_coords = dataarray_cut.coords["phi"].values
        # rad -> deg
        dataarray_cut.S.swap_angle_unit()
        phi_coords = dataarray_cut.coords["phi"].values
        np.testing.assert_array_almost_equal(phi_coords[0:6], [12.7, 12.8, 12.9, 13.0, 13.1, 13.2])
        assert (
            dataarray_cut.coords["chi"]
            == dataarray_cut.attrs["chi_offset"]
            == np.rad2deg(-0.10909301748228785)
        )
        assert dataarray_cut.attrs["chi_offset"] == np.rad2deg(-0.10909301748228785)
        assert dataarray_cut.S.angle_unit == "Degrees"
        # deg -> rad
        dataarray_cut.S.swap_angle_unit()

        np.testing.assert_array_almost_equal(
            dataarray_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataarray_cut.S.angle_unit == "Radians"
        #
        dataarray_cut.attrs["angle_unit"] = "Rad."
        with pytest.raises(TypeError):
            dataarray_cut.S.swap_angle_unit()

    def test_for_is_slit_vertical(self, dataarray_cut: xr.DataArray) -> None:
        """Test for is_slit_vertical (DataArray version)."""
        assert dataarray_cut.S.is_slit_vertical is False
        dataarray_cut.coords["alpha"] = np.pi / 2
        assert dataarray_cut.S.is_slit_vertical is True
        #
        dataarray_cut.S.swap_angle_unit()
        assert dataarray_cut.S.is_slit_vertical is True


class TestAngleUnitForDataset:
    """Test class for angle_unit for DataSet."""

    def test_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle unit property for Dataset."""
        assert dataset_cut.S.angle_unit == "Radians"

    def test_setter_of_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle_unit setter. (Dataset)."""
        dataset_cut.S.angle_unit = "Degrees"
        assert dataset_cut.S.angle_unit == "Degrees" == dataset_cut.attrs["angle_unit"]
        for spectrum in dataset_cut.S.spectra:
            assert spectrum.S.angle_unit == "Degrees"

    def test_swap_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for swap_angle_unit (Dataset version)."""
        original_phi_coords = dataset_cut.coords["phi"].values
        # rad -> deg
        dataset_cut.S.swap_angle_unit()
        phi_coords = dataset_cut.coords["phi"].values
        np.testing.assert_array_almost_equal(phi_coords[0:6], [12.7, 12.8, 12.9, 13.0, 13.1, 13.2])
        assert (
            dataset_cut.coords["chi"]
            == dataset_cut.attrs["chi_offset"]
            == np.rad2deg(-0.10909301748228785)
        )
        assert dataset_cut.attrs["chi_offset"] == np.rad2deg(-0.10909301748228785)
        assert dataset_cut.S.angle_unit == "Degrees"
        for spectrum in dataset_cut.S.spectra:
            assert (
                spectrum.coords["chi"]
                == spectrum.attrs["chi_offset"]
                == np.rad2deg(-0.10909301748228785)
            )
            np.testing.assert_array_almost_equal(
                spectrum.coords["phi"][0:6],
                [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
            )

        # deg -> rad
        dataset_cut.S.swap_angle_unit()
        np.testing.assert_array_almost_equal(
            dataset_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataset_cut.S.angle_unit == "Radians"

        for spectrum in dataset_cut.S.spectra:
            np.testing.assert_array_almost_equal(
                spectrum.coords["phi"].values[0:6],
                original_phi_coords[0:6],
            )

        # Exception test
        dataset_cut.attrs["angle_unit"] = "Rad."
        with pytest.raises(TypeError):
            dataset_cut.S.swap_angle_unit()

    def test_for_is_slit_vertical(self, dataset_cut: xr.Dataset) -> None:
        """Test for is_slit_vertical (Dataset version)."""
        assert dataset_cut.S.is_slit_vertical is False
        dataset_cut.coords["alpha"] = np.pi / 2
        for spectrum in dataset_cut.S.spectra:
            spectrum.coords["alpha"] = np.pi / 2
        assert dataset_cut.S.is_slit_vertical is True
        #
        dataset_cut.S.swap_angle_unit()
        assert dataset_cut.S.is_slit_vertical is True


def test_experimental_conditions(dataset_cut: xr.Dataset) -> None:
    """Unit test for experimental_conditions property.

    [TODO:description]

    Args:
        dataset_cut: [TODO:description]

    Returns:
        [TODO:description]
    """
    assert dataset_cut.S.experimental_conditions == {
        "hv": 5.93,
        "polarization": None,
        "temperature": None,
    }


def test_predicates() -> None:
    """Test for predicates.

    1. is_subtracted
    2. is_spatial
    3. is_kspace
    4. is_slit_vertical
    6. is_differentiated


    :return:
    """


def test_location_and_endstation(dataset_cut: xr.Dataset) -> None:
    """Unit test for endstation property."""
    assert dataset_cut.S.endstation == "ALG-MC"


def test_spectrometer() -> None:
    """Test for spectrometer.

    [TODO:description]

    """


def test_attribute_normalization() -> None:
    """Test for attribute normalization.

    1. t0
    2. hv
    3. manipulator/sample location values
    4. beamline settings
    ...

    A full list of these is available at the doc site under
    the description of the data model.

    This is at:
    https://arpes.readthedocs.io/spectra

    :return:
    """


def test_id_and_identification_attributes() -> None:
    """Tests for id attributes.

    1. id
    2. original_id
    3. scan_name
    4. label
    5. original_parent_scan_name

    :return:
    """


def test_dataset_attachment() -> None:
    """Tests for dataet attachment.

    1.  scan_row
    2. df_index
    3. df_after
    4. df_until_type
    5. referenced_scans

    :return:
    """


def test_sample_position() -> None:
    """Test for sample_positon.

    This is also an opportunity to test the dimension/axis
    conventions on each beamline
    """


def test_raveling() -> None:
    """Tests ravel and meshgrid."""


def test_fp_mapping() -> None:
    """Tests `map_axes` and `map`."""


# fit utilities
def test_attribute_accessors() -> None:
    """Tests.

    1. .p
    2. .s

    :return:
    """
