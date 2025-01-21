"""Test for basic data loading."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions  # pylint: disable=unused-import, redefined-outer-name  # noqa: F401
from arpes.utilities.conversion import convert_to_kspace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import Incomplete


def pytest_generate_tests(metafunc: Incomplete) -> Incomplete:
    """[TODO:summary].

    [TODO:description]

    Args:
        metafunc ([TODO:type]): [TODO:description]
    """
    idlist = []
    argvalues = []

    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])

    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


TOBETESTED = dict[str, str | dict[str, str | None | float | dict[str, str | None | float]]]


class TestMetadata:
    """Tests metadata normalization conventions."""

    data = None

    scenarios: ClassVar = [
        # Lanzara Group "Main Chamber"
        (
            "main_chamber_load_cut",
            {
                "file": "basic/main_chamber_cut_0.fits",
                "expected": {
                    "scan_info": {
                        "time": "1:45:34 pm",
                        "date": "2/3/2016",
                        "type": None,
                        "spectrum_type": "cut",
                        "experimenter": None,
                        "sample": None,
                    },
                    "experiment_info": {
                        "temperature": np.nan,
                        "temperature_cryotip": np.nan,
                        "pressure": np.nan,
                        "polarization": (np.nan, np.nan),
                        "photon_flux": np.nan,
                        "photocurrent": np.nan,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer_detail": {
                            "analyzer_type": "hemispherical",
                            "analyzer_radius": 150,
                            "analyzer_name": "Specs PHOIBOS 150",
                            "parallel_deflectors": False,
                            "perpendicular_deflectors": False,
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": None,
                        "lens_mode_name": "WideAngleMode:40V",
                        "acquisition_mode": None,
                        "pass_energy": np.nan,
                        "slit_shape": None,
                        "slit_width": np.nan,
                        "slit_number": np.nan,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": np.nan,
                        "work_function": 4.401,
                    },
                    "daq_info": {
                        "daq_type": None,
                        "region": None,
                        "region_name": None,
                        "prebinning": {"eV": 2, "phi": 1},
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "low_energy": None,
                            "high_energy": None,
                            "n_sweeps": None,
                            "step": None,
                        },
                        "frames_per_slice": 500,
                        "frame_duration": np.nan,
                        "center_energy": np.nan,
                    },
                    "laser_info": {
                        "pump_wavelength": np.nan,
                        "pump_energy": np.nan,
                        "pump_fluence": np.nan,
                        "pump_pulse_energy": np.nan,
                        "pump_spot_size": (np.nan, np.nan),
                        "pump_profile": None,
                        "pump_linewidth": np.nan,
                        "pump_duration": np.nan,
                        "pump_polarization": (np.nan, np.nan),
                        "probe_wavelength": np.nan,
                        "probe_energy": 5.93,
                        "probe_fluence": np.nan,
                        "probe_pulse_energy": np.nan,
                        "probe_spot_size": (np.nan, np.nan),
                        "probe_profile": None,
                        "probe_linewidth": 0.015,
                        "probe_duration": np.nan,
                        "probe_polarization": (np.nan, np.nan),
                        "repetition_rate": np.nan,
                    },
                    "sample_info": {
                        "id": None,
                        "sample_name": None,
                        "source": None,
                        "reflectivity": np.nan,
                    },
                },
            },
        ),
        (
            "merlin_load_cut",
            {
                "file": "basic/MERLIN_8.pxt",
                "expected": {
                    "scan_info": {
                        "time": "09:52:10 AM",
                        "date": "07/05/2017",
                        "type": None,
                        "spectrum_type": "cut",
                        "experimenter": "Jonathan",
                        "sample": "LaSb_3",
                    },
                    "experiment_info": {
                        "temperature": 21.75,
                        "temperature_cryotip": 21.43,
                        "pressure": 3.11e-11,
                        "polarization": (0, 0),
                        "photon_flux": 2.652,
                        "photocurrent": np.nan,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer_detail": {
                            "analyzer_name": "Scienta R8000",
                            "parallel_deflectors": False,
                            "perpendicular_deflectors": False,
                            "analyzer_radius": np.nan,
                            "analyzer_type": "hemispherical",
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": None,
                        "lens_mode_name": "Angular30",
                        "acquisition_mode": "swept",
                        "pass_energy": 20,
                        "slit_shape": "curved",
                        "slit_width": 0.5,
                        "slit_number": 7,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": 1550,
                        "work_function": 4.401,
                    },
                    "beamline_info": {
                        "hv": 90.0,
                        "beam_current": 500.761,
                        "linewidth": np.nan,
                        "photon_polarization": (0, 0),
                        "entrance_slit": 50.1,
                        "exit_slit": 50.1,
                        "undulator_info": {
                            "harmonic": 2,
                            "type": "elliptically_polarized_undulator",
                            "gap": 41.720,
                            "z": 0.0,
                            "polarization": 0,
                        },
                        "repetition_rate": 5e8,
                        "monochromator_info": {
                            "grating_lines_per_mm": np.nan,
                        },
                    },
                    "daq_info": {
                        "daq_type": None,
                        "region": "Swept_VB4",
                        "region_name": "Swept_VB4",
                        "prebinning": {},
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "n_sweeps": 4,
                            "step": 0.002,
                            "low_energy": 88.849,
                            "high_energy": 90.199,
                        },
                        "frames_per_slice": np.nan,
                        "frame_duration": np.nan,
                        "center_energy": 87.5,
                    },
                    "sample_info": {
                        "id": None,
                        "sample_name": "LaSb_3",
                        "source": None,
                        "reflectivity": np.nan,
                    },
                },
            },
        ),
        (
            "maestro_load_cut",
            {
                "file": "basic/MAESTRO_12.fits",
                "expected": {
                    "scan_info": {
                        "time": "7:08:42 pm",
                        "date": "10/11/2018",
                        "type": "XY Scan",
                        "spectrum_type": "spem",
                        "experimenter": None,
                        "sample": None,
                    },
                    "experiment_info": {
                        "temperature": np.nan,
                        "temperature_cryotip": np.nan,
                        "pressure": np.nan,
                        "polarization": (np.nan, np.nan),
                        "photon_flux": np.nan,
                        "photocurrent": np.nan,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer_detail": {
                            "analyzer_type": "hemispherical",
                            "analyzer_radius": np.nan,
                            "analyzer_name": "Scienta R4000",
                            "parallel_deflectors": False,
                            "perpendicular_deflectors": True,
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": None,
                        "lens_mode_name": "Angular30",
                        "acquisition_mode": None,
                        "pass_energy": 50,
                        "slit_shape": "curved",
                        "slit_width": 0.5,
                        "slit_number": 7,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": np.nan,
                        "work_function": 4.401,
                    },
                    "beamline_info": {
                        "hv": pytest.approx(125, 1e-2),
                        "linewidth": np.nan,
                        "beam_current": pytest.approx(500.44, 1e-2),
                        "photon_polarization": (np.nan, np.nan),
                        "repetition_rate": 5e8,
                        "entrance_slit": None,
                        "exit_slit": None,
                        "undulator_info": {
                            "harmonic": 1,
                            "type": "elliptically_polarized_undulator",
                            "gap": None,
                            "z": None,
                            "polarization": None,
                        },
                        "monochromator_info": {
                            "grating_lines_per_mm": 600,
                        },
                    },
                    "daq_info": {
                        "daq_type": "XY Scan",
                        "region": None,
                        "region_name": None,
                        "prebinning": {
                            "eV": 2,
                        },
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "low_energy": None,
                            "high_energy": None,
                            "n_sweeps": None,
                            "step": None,
                        },
                        "frames_per_slice": 10,
                        "frame_duration": np.nan,
                        "center_energy": 33.2,
                    },
                    "sample_info": {
                        "id": None,
                        "sample_name": None,
                        "source": None,
                        "reflectivity": np.nan,
                    },
                },
            },
        ),
        (
            "uranos_load_cut",
            {
                "file": "basic/Uranos_cut.pxt",
                "expected": {
                    "scan_info": {
                        "time": "22:34:26",
                        "date": "2024-05-17",
                        "type": None,
                        "spectrum_type": "cut",
                        "experimenter": None,
                        "sample": "MK",
                    },
                    "experiment_info": {
                        "temperature": np.nan,
                        "temperature_cryotip": np.nan,
                        "pressure": np.nan,
                        "polarization": (np.nan, np.nan),
                        "photon_flux": np.nan,
                        "photocurrent": np.nan,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer_detail": {
                            "analyzer_name": "DA30L",
                            "parallel_deflectors": True,
                            "perpendicular_deflectors": True,
                            "analyzer_type": "hemispherical",
                            "analyzer_radius": np.nan,
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": "DA30L_01",
                        "lens_mode_name": None,
                        "acquisition_mode": "Swept",
                        "pass_energy": 10,
                        "slit_shape": None,
                        "slit_width": np.nan,
                        "slit_number": np.nan,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": np.nan,
                        "work_function": 4.401,
                    },
                    "beamline_info": {
                        "hv": pytest.approx(20, 1e-2),
                        "linewidth": np.nan,
                        "undulator_info": {
                            "harmonic": None,
                            "type": None,
                            "gap": None,
                            "z": None,
                            "polarization": None,
                        },
                        "repetition_rate": np.nan,
                        "beam_current": np.nan,
                        "photon_polarization": (np.nan, np.nan),
                        "entrance_slit": None,
                        "exit_slit": None,
                        "monochromator_info": {
                            "grating_lines_per_mm": np.nan,
                        },
                    },
                    "daq_info": {
                        "daq_type": None,
                        "region": None,
                        "region_name": None,
                        "center_energy": np.nan,
                        "prebinning": {},
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "low_energy": 12,
                            "high_energy": 15.8,
                            "n_sweeps": 67,
                            "step": 0.004,
                        },
                        "frames_per_slice": np.nan,
                        "frame_duration": np.nan,
                    },
                    "sample_info": {
                        "id": None,
                        "sample_name": "MK",
                        "source": None,
                        "reflectivity": np.nan,
                    },
                },
            },
        ),
        (
            "dsnp_umcs_load_cut",
            {
                "file": "basic/DSNP_UMCS_cut.xy",
                "expected": {
                    "scan_info": {
                        "time": "13:08:08",
                        "date": "11/28/24",
                        "type": None,
                        "spectrum_type": "cut",
                        "experimenter": None,
                        "sample": None,
                    },
                    "experiment_info": {
                        "temperature": np.nan,
                        "temperature_cryotip": np.nan,
                        "pressure": np.nan,
                        "polarization": (np.nan, np.nan),
                        "photon_flux": np.nan,
                        "photocurrent": np.nan,
                        "probe": None,
                        "probe_detail": None,
                        "analyzer_detail": {
                            "analyzer_name": "Specs PHOIBOS 150",
                            "parallel_deflectors": False,
                            "perpendicular_deflectors": False,
                            "analyzer_type": "hemispherical",
                            "analyzer_radius": 150,
                        },
                    },
                    "analyzer_info": {
                        "lens_mode": "LowAngularDispersion:400V",
                        "lens_mode_name": None,
                        "acquisition_mode": None,
                        "pass_energy": 15.0,
                        "slit_shape": None,
                        "slit_width": 0.5,
                        "slit_number": 2,
                        "lens_table": None,
                        "analyzer_type": "hemispherical",
                        "mcp_voltage": 1450.0,
                        "work_function": 4.32,
                    },
                    "beamline_info": {
                        "hv": 21.2182,
                        "linewidth": np.nan,
                        "photon_polarization": (np.nan, np.nan),
                        "undulator_info": {
                            "gap": None,
                            "z": None,
                            "harmonic": None,
                            "polarization": None,
                            "type": None,
                        },
                        "repetition_rate": np.nan,
                        "beam_current": np.nan,
                        "entrance_slit": None,
                        "exit_slit": None,
                        "monochromator_info": {"grating_lines_per_mm": np.nan},
                    },
                    "daq_info": {
                        "daq_type": None,
                        "region": None,
                        "region_name": None,
                        "center_energy": np.nan,
                        "prebinning": {},
                        "trapezoidal_correction_strategy": None,
                        "dither_settings": None,
                        "sweep_settings": {
                            "high_energy": None,
                            "low_energy": None,
                            "n_sweeps": None,
                            "step": None,
                        },
                        "frames_per_slice": np.nan,
                        "frame_duration": np.nan,
                    },
                    "sample_info": {
                        "id": None,
                        "sample_name": None,
                        "source": None,
                        "reflectivity": np.nan,
                    },
                },
            },
        ),
    ]

    def test_load_file_and_basic_attributes(
        self,
        sandbox_configuration: Incomplete,
        file: str,
        expected: dict[str, str | None | dict[str, float]],
    ) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            sandbox_configuration ([TODO:type]): [TODO:description]
            file ([TODO:type]): [TODO:description]
            expected ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        data = sandbox_configuration.load(file)
        assert isinstance(data, xr.Dataset)

        for k, v in expected.items():
            metadata = getattr(data.S, k)
            assert k
            assert metadata == v


class TestBasicDataLoading:
    """Tests procedures/plugins for loading basic data."""

    data = None

    scenarios: ClassVar[list[Incomplete]] = [
        # Lanzara Group "Main Chamber"
        (
            "main_chamber_load_cut",
            {
                "file": "basic/main_chamber_cut_0.fits",
                "expected": {
                    "dims": ["phi", "eV"],
                    "coords": {
                        "phi": [0.22165, 0.63879, 0.001745],
                        "eV": [-0.42558, 0.130235, 0.0023256],
                        "alpha": 0,
                        "x": -0.770435,
                        "y": 34.75,
                        "z": -3.4e-5,
                        "chi": -6.25 * np.pi / 180,
                    },
                    "offset_coords": {
                        "phi": 0.22165 - 0.405,
                        "beta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "main_chamber_load_map",
            {
                "file": "basic/main_chamber_map_1.fits",
                "expected": {
                    "dims": ["beta", "phi", "eV"],
                    "coords": {
                        "beta": [-0.314159, 0.61086, 0.03426],
                        "phi": [0.15184, 0.862193, 0.001745],
                        "eV": [-1.369977, 0.431409, 0.0046189],
                        "x": 2.99,
                        "y": 41.017,
                        "z": -0.03104,
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0.15184 - 0.405,
                        "beta": -0.314159,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "main_chamber_load_multi_region",
            {
                "file": "basic/main_chamber_PHONY_2.fits",
                "expected": {
                    "dims": ["phi", "eV"],
                    "coords": {
                        "phi": [0.22165, 0.63879, 0.001745],
                        "eV": [-0.42558, 0.130235, 0.0023256],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0.22165 - 0.405,
                        "beta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "main_chamber_load_single_cycle",
            {
                "file": "basic/main_chamber_PHONY_3.fits",
                "expected": {
                    "dims": ["phi", "eV"],
                    "coords": {
                        "phi": [0.22165, 0.63879, 0.001745],
                        "eV": [-0.42558, 0.130235, 0.0023256],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0.22165 - 0.405,
                        "beta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        # Lanzara Group "Spin-ToF"
        # ('stof_load_edc', {
        # }),
        # ('stof_load_spin_edc', {
        # }),
        # ('stof_load_map', {
        # }),
        # ('stof_load_spin_map', {
        # }),
        # ALS Beamline 4 "MERLIN" / SES
        (
            "merlin_load_cut",
            {
                "file": "basic/MERLIN_8.pxt",
                "expected": {
                    "dims": ["eV", "phi"],
                    "coords": {
                        "phi": [-0.29103, 0.34335, 0.00081749],
                        "eV": [-2.5, 0.2001, 0.002],
                        "alpha": np.pi / 2,
                    },
                    "offset_coords": {"phi": -0.29103, "theta": 0.1043, "chi": 0},
                },
            },
        ),
        (
            "merlin_load_xps",
            {
                "file": "basic/MERLIN_9.pxt",
                "expected": {
                    "dims": ["eV"],
                    "coords": {
                        "eV": [-55, 0.99915, 0.0999],
                        "alpha": np.pi / 2,
                        "chi": -107.09 * np.pi / 180,
                    },
                    "offset_coords": {"phi": 0, "theta": 0.002 * np.pi / 180, "chi": 0},
                },
            },
        ),
        (
            "merlin_load_map",
            {
                "file": "basic/MERLIN_10_S001.pxt",
                "expected": {
                    "dims": ["theta", "eV", "phi"],
                    "coords": {
                        "theta": [-0.209439, -0.200713, 0.008726],
                        "phi": [-0.29103, 0.34335, 0.00081749],
                        "eV": [-1.33713, 0.33715, 0.00159],
                        "alpha": np.pi / 2,
                    },
                    "offset_coords": {"phi": -0.29103, "theta": -0.209439, "chi": 0},
                },
            },
        ),
        (
            "merlin_load_hv",
            {
                "file": "basic/MERLIN_11_S001.pxt",
                "expected": {
                    "dims": ["hv", "eV", "phi"],
                    "coords": {
                        "hv": [108, 110, 2],
                        "phi": [-0.29103, 0.34335, 0.00081749],
                        "eV": [-1.33911, 0.34312, 0.00159],
                        "alpha": np.pi / 2,
                    },
                    "offset_coords": {"phi": -0.29103, "theta": -0.999 * np.pi / 180, "chi": 0},
                },
            },
        ),
        # ALS Beamline 7 "MAESTRO"
        (
            "maestro_load_cut",
            {
                "file": "basic/MAESTRO_12.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [4.961, 5.7618, 0.04],
                        "x": [0.86896, 1.6689, 0.04],
                        "eV": [-35.478, -31.2837, 0.00805],
                        "z": -0.4,
                        "alpha": 0,
                    },
                    "offset_coords": {"phi": 0, "theta": 0, "chi": 0},
                },
            },
        ),
        (
            "maestro_load_xps",
            {
                "file": "basic/MAESTRO_13.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [-0.92712, -0.777122, 0.010714],
                        "x": [0.42983, 0.57983, 0.010714],
                        "eV": [32.389, 39.9296, 0.011272],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0,
                        "theta": 10.008 * np.pi / 180,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_map",
            {
                "file": "basic/MAESTRO_PHONY_14.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [4.961, 5.7618, 0.04],
                        "x": [0.86896, 1.6689, 0.04],
                        "eV": [-35.478, -31.2837, 0.00805],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0,
                        "theta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_hv",
            {
                "file": "basic/MAESTRO_PHONY_15.fits",
                "expected": {
                    "dims": ["y", "x", "eV"],
                    "coords": {
                        "y": [4.961, 5.7618, 0.04],
                        "x": [0.86896, 1.6689, 0.04],
                        "eV": [-35.478, -31.2837, 0.00805],
                        "alpha": 0,
                    },
                    "offset_coords": {
                        "phi": 0,
                        "theta": 0,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_multi_region",
            {
                "file": "basic/MAESTRO_16.fits",
                "expected": {
                    "dims": ["eV"],
                    "coords": {"eV": [-1.5, 0.50644, 0.0228], "alpha": 0},
                    "offset_coords": {
                        "phi": 0,
                        "theta": 10.062 * np.pi / 180,
                        "chi": 0,
                    },
                },
            },
        ),
        (
            "maestro_load_nano_arpes_hierarchical_manipulator",
            {
                "file": "basic/MAESTRO_nARPES_focus_17.fits",
                "expected": {
                    "dims": ["optics_insertion", "y", "eV"],
                    "coords": {
                        "eV": [-35.16, -28.796, 0.01095],
                        "optics_insertion": [-100, 100, 10],
                        "y": [935.67, 935.77, -0.005],
                        "alpha": np.pi / 2,
                    },
                    "offset_coords": {
                        "phi": -0.4,
                        "theta": 1.4935e-6,
                        "chi": 0,
                    },
                },
            },
        ),
        # Solaris, Uranos beamline
        (
            "uranos_load_cut",
            {
                "file": "basic/Uranos_cut.pxt",
                "expected": {
                    "dims": ["eV", "phi"],
                    "coords": {
                        "x": 1.1,
                        "y": -2.3,
                        "z": 0.0,
                        "beta": 0.0,
                        "theta": 0.27932,
                        "alpha": np.deg2rad(90),
                        "psi": 0.0,
                        "eV": [-3.619399, 0.1806009, 0.003999],
                        "phi": [-0.28633, 0.26867, 0.0008409],
                    },
                    "offset_coords": {},
                },
            },
        ),
        # DSNP_UMCS, cut data
        (
            "dsnp_umcs_load_cut",
            {
                "file": "basic/DSNP_UMCS_cut.xy",
                "expected": {
                    "dims": ["eV", "phi"],
                    "coords": {
                        "x": 78.0,
                        "y": 0.5,
                        "z": 2.5,
                        "beta": 0.0,
                        "theta": 0.0,
                        "alpha": np.deg2rad(90),
                        "psi": 0.0,
                        "eV": [-1.5, 0.134188, 0.012016],
                        "phi": [-0.120683, 0.120683, 0.00298],
                    },
                    "offset_coords": {},
                },
            },
        ),
    ]

    def test_load_file_and_basic_attributes(
        self,
        sandbox_configuration: Incomplete,
        file: str,
        expected: dict[str, Any],
    ) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            sandbox_configuration ([TODO:type]): [TODO:description]
            file ([TODO:type]): [TODO:description]
            expected ([TODO:type]): [TODO:description]
        """
        data = sandbox_configuration.load(file)
        assert isinstance(data, xr.Dataset)

        # assert basic dataset attributes
        for attr in ["location"]:
            assert attr in data.attrs

        # assert that all necessary coordinates are present
        necessary_coords = {"phi", "psi", "alpha", "chi", "beta", "theta", "x", "y", "z", "hv"}
        for necessary_coord in necessary_coords:
            assert necessary_coord in data.coords

        # assert basic spectrum attributes
        for attr in ["hv", "location"]:
            if attr == "hv" and (
                data.S.spectrum.attrs.get("spectrum_type") == "hv_map" or len(data.S.spectra) > 1
            ):
                continue
            assert attr in data.S.spectrum.attrs

        # assert dimensions
        assert list(data.S.spectra[0].dims) == expected["dims"]

        # assert coordinate shape
        by_dims = data.S.spectra[0].dims
        ranges = [
            [
                pytest.approx(data.coords[d].min().item(), 1e-3),
                pytest.approx(data.coords[d].max().item(), 1e-3),
                pytest.approx(data.G.stride(generic_dim_names=False)[d], 1e-3),
            ]
            for d in by_dims
        ]

        assert list(zip(by_dims, ranges, strict=True)) == list(
            zip(by_dims, [expected["coords"][d] for d in by_dims], strict=True),
        )
        for k, v in expected["coords"].items():
            if isinstance(v, float):
                assert k
                assert pytest.approx(data.coords[k].item(), 1e-3) == v

        def safefirst(x: Iterable[float]) -> float:
            with contextlib.suppress(TypeError, IndexError):
                return x[0]

            with contextlib.suppress(AttributeError):
                return x.item()

            return x

        for k in expected["offset_coords"]:
            offset = safefirst(data.S.spectra[0].S.lookup_offset_coord(k))
            assert k
            assert pytest.approx(offset, 1e-3) == expected["offset_coords"][k]

        kspace_data = convert_to_kspace(data.S.spectra[0])
        assert isinstance(kspace_data, xr.DataArray)
