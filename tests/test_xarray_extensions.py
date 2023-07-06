import numpy as np
import pytest
import xarray as xr

from arpes.io import example_data


@pytest.fixture()
def dataset_cut() -> xr.Dataset:
    return example_data.cut


@pytest.fixture()
def dataarray_cut() -> xr.DataArray:
    return example_data.cut.spectrum


@pytest.mark.skip()
class TestAngleUnitForDataset:
    """Test class for angle_unit for DataSet."""

    def test_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle unit property for Dataset."""
        assert dataset_cut.angle_unit == "Radians"


class TestAngleUnitforDataArray:
    """Test class for angle_unit for DataArray."""

    def test_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle unit property for DataArray."""
        assert dataarray_cut.S.angle_unit == "Radians"

    def test_setter_of_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle_unit setter."""
        dataarray_cut.S.angle_unit = "Degrees"
        assert dataarray_cut.S.angle_unit == "Degrees"

    def test_swap_angle_unit(self, dataarray_cut: xr.DataArray):
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


def test_experimental_conditions():
    pass


def test_predicates():
    """Namely:

    1. is_subtracted
    2. is_spatial
    3. is_kspace
    4. is_slit_vertical
    6. is_differentiated


    :return:
    """


def test_location_and_endstation():
    pass


def test_spectrometer():
    pass


def test_attribute_normalization():
    """1. t0
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


def test_spectrum_type():
    # TODO DEDUPE THIS
    pass


def test_transposition():
    pass


def test_select_around_data():
    pass


def test_select_around():
    pass


def test_shape():
    pass


def test_id_and_identification_attributes():
    """Tests:

    1. id
    2. original_id
    3. scan_name
    4. label
    5. original_parent_scan_name

    :return:
    """


def test_dataset_attachment():
    """Tests:

    1. scan_row
    2. df_index
    3. df_after
    4. df_until_type
    5. referenced_scans

    :return:
    """


def test_sum_other():
    pass


def test_reference_settings():
    pass


def test_beamline_settings():
    pass


def test_spectrometer_settings():
    pass


def test_sample_position():
    """This is also an opportunity to test the dimension/axis
    conventions on each beamline
    :return:
    """


def test_full_coords():
    pass


def test_cut_nan_coords():
    pass


def test_drop_nan():
    pass


def test_scale_coords():
    pass


def test_transform_coords():
    pass


def test_coordinatize():
    pass


def test_raveling():
    """Tests ravel and meshgrid
    :return:
    """


def test_to_arrays():
    pass


# Functional programming utilities
def test_iterate_axis():
    pass


def test_fp_mapping():
    """Tests `map_axes` and `map`
    :return:
    """


def test_enumerate_iter_coords():
    pass


def test_iter_coords():
    pass


def test_dataarray_range():
    pass


def test_stride():
    pass


# shifting
def test_shift_by():
    pass


def test_shift_coords():
    pass


# fit utilities
def test_attribute_accessors():
    """Tests.

    1. .p
    2. .s

    :return:
    """


def test_model_evaluation():
    """:return:"""


def test_param_as_dataset():
    pass


def test_parameter_names():
    pass


def test_spectrum_and_spectra_selection():
    pass


def test_degrees_of_freedom():
    # TODO decide how to de-dedupe axes that have been suffixed with
    # spectrum number for multi-region scans
    pass
