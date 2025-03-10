"""Contains calibrations and information for spectrometer resolution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

# all resolutions are given by (photon energy, entrance slit, exit slit size)
from arpes.constants import K_BOLTZMANN_MEV_KELVIN
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from arpes._typing import XrTypes

__all__ = ("total_resolution_estimate",)


# all analyzer dimensions are given in millimeters for convenience as this
# is how slit sizes are typically reported
def r8000(slits: list[float]) -> dict[str, Any]:
    return {
        "type": "HEMISPHERE",
        "slits": slits,
        "radius": 200,
        "angle_resolution": np.deg2rad(0.1),
    }


def analyzer_resolution(
    analyzer_information: dict[str, Any],
    slit_width: float | None = None,
    slit_number: int | None = None,
    pass_energy: float = 10,
) -> float:
    """Estimates analyzer resolution from slit dimensioons passgenergy, and analyzer radius.

    Args:
        analyzer_information: The analyzer specification containing slit information.
        slit_width: The width of the slit in mm.
        slit_number: The slit number, if the slit width was not provided.
        pass_energy: The pass energy used in the analyzer, in eV.

    Returns:
        The resolution of the analyzer for a given slit-pass energy configuration.
    """
    if slit_width is None:
        slit_width = analyzer_information["slits"][slit_number]

    return 1000 * pass_energy * (slit_width / (2 * analyzer_information["radius"]))


SPECTROMETER_INFORMATION = {"BL403": r8000([0.05, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.8])}

MERLIN_BEAMLINE_RESOLUTION: dict[str, dict[tuple[float, tuple[float, float]], float]] = {
    "LEG": {
        # 40 um by 40 um slits
        (25.0, (40.0, 40.0)): 9.5,
        (30, (40, 40)): 13.5,
        (35, (40, 40)): 22.4,
    },
    "HEG": {
        # 30 um by 30 um slits
        (30, (30, 30)): 5.2,
        (35, (30, 30)): 5.4,
        (40, (30, 30)): 5.0,
        (45, (30, 30)): 5.5,
        (50, (30, 30)): 5.4,
        (55, (30, 30)): 5.5,
        (60, (30, 30)): 6.4,
        (65, (30, 30)): 6.5,
        (70, (30, 30)): 7.7,
        (75, (30, 30)): 9,
        (80, (30, 30)): 8.6,
        (90, (30, 30)): 12.4,
        (100, (30, 30)): 21.6,
        (110, (30, 30)): 35.4,
        # 50um by 50um
        (60, (50, 50)): 8.2,
        (70, (50, 50)): 10.2,
        (80, (50, 50)): 12.6,
        (90, (50, 50)): 16.5,
        # 60um by 80um
        (60, (60, 80)): 9.2,
        (70, (60, 80)): 13.0,
        (80, (60, 80)): 16.5,
        (90, (60, 80)): 22.0,
        # 90um by 140um
        (30, (90, 140)): 7.3,
        (40, (90, 140)): 9,
        (50, (90, 140)): 11.9,
        (60, (90, 140)): 16.2,
        (70, (90, 140)): 21.4,
        (80, (90, 140)): 25.8,
        (90, (90, 140)): 34,
        (100, (90, 140)): 45,
        (110, (90, 140)): 60,
    },
    # second order from the high density grating
    "HEG-2": {
        # 30um by 30um
        (90, (30, 30)): 8,
        (100, (30, 30)): 9,
        (110, (30, 30)): 9.6,
        (120, (30, 30)): 9.6,
        (130, (30, 30)): 12,
        (140, (30, 30)): 15,
        (150, (30, 30)): 13,
        # 50um by 50um
        (90, (50, 50)): 10.3,
        (100, (50, 50)): 10.5,
        (110, (50, 50)): 13.2,
        (120, (50, 50)): 14,
        (130, (50, 50)): 19,
        (140, (50, 50)): 22,
        (150, (50, 50)): 22,
        # 60um by 80um
        (90, (60, 80)): 12.8,
        (100, (60, 80)): 15,
        (110, (60, 80)): 16.4,
        (120, (60, 80)): 19,
        (130, (60, 80)): 27,
        (140, (60, 80)): 30,
        # 90um by 140um
        (90, (90, 140)): 19,
        (100, (90, 140)): 21,
        (110, (90, 140)): 28,
        (120, (90, 140)): 31,
        (130, (90, 140)): 37,
        (140, (90, 140)): 41,
        (150, (90, 140)): 49,
    },
}

ENDSTATIONS_BEAMLINE_RESOLUTION = {
    "BL403": MERLIN_BEAMLINE_RESOLUTION,
}


def analyzer_resolution_estimate(
    data: xr.DataArray,
    *,
    meV: bool = False,  # noqa: N803
) -> float:
    """Estimates the energy resolution of the analyzer.

    For hemispherical analyzers, this can be determined by the slit
    and pass energy settings.

    Args:
        data(DataType): The data to estimate for. Used to extract spectrometer info.
        meV (bool): If True, returns resolution in meV units.

    Returns:
        The resolution in eV units.
    """
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    endstation = data_array.S.endstation
    spectrometer_info = SPECTROMETER_INFORMATION[endstation]

    spectrometer_settings = data.S.spectrometer_settings

    return analyzer_resolution(
        spectrometer_info,
        slit_number=spectrometer_settings["slit"],
        pass_energy=spectrometer_settings["pass_energy"],
    ) * (1 if meV else 0.001)


def energy_resolution_from_beamline_slit(
    table: dict[tuple[float, tuple[float, float]], float],
    photon_energy: float,
    exit_slit_size: tuple[float, float],
) -> float:
    """Calculates the energy resolution contribution from the beamline slits.

    Assumes an exact match on the photon energy, though that interpolation
    could also be pulled into here...

    Args:
        table: Beamline specific calibration table.
        photon_energy: The photon energy used.
        exit_slit_size: The exit slit size used by the beamline.

    Returns:
        The energy broadening in eV.
    """
    by_slits: dict[tuple[float, float], float] = {
        k[1]: v for k, v in table.items() if k[0] == photon_energy
    }
    if exit_slit_size in by_slits:
        return by_slits[exit_slit_size]

    slit_area: float = exit_slit_size[0] * exit_slit_size[1]
    by_area = {int(k[0] * k[1]): v for k, v in by_slits.items()}

    if len(by_area) == 1:
        return next(iter(by_area.values())) * slit_area / (next(iter(by_area.keys())))

    try:
        low = max(k for k in by_area if k <= slit_area)
        high = min(k for k in by_area if k >= slit_area)
    except ValueError:
        if slit_area > max(by_area.keys()):
            # use the largest and second largest
            high = max(by_area.keys())
            low = max(k for k in by_area if k < high)
        else:
            # use the smallest and second smallest
            low = min(by_area.keys())
            high = min(k for k in by_area if k > low)

    return by_area[low] + (by_area[high] - by_area[low]) * (slit_area - low) / (high - low)


def beamline_resolution_estimate(
    data: xr.DataArray,
    *,
    meV: bool = False,  # noqa: N803
) -> float:
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    resolution_table: dict[str, dict[tuple[float, tuple[float, float]], float]] = (
        ENDSTATIONS_BEAMLINE_RESOLUTION[data_array.S.endstation]
    )

    if isinstance(next(iter(resolution_table.keys())), str):
        # need grating information
        settings = data_array.S.beamline_settings
        resolution_table_selected: dict[tuple[float, tuple[float, float]], float] = (
            resolution_table[settings["grating"]]
        )
        all_keys = list(resolution_table_selected.keys())
        hvs = {k[0] for k in all_keys}

        low_hv = max(hv for hv in hvs if hv < settings["hv"])
        high_hv = min(hv for hv in hvs if hv >= settings["hv"])

        slit_size = (
            settings["entrance_slit"],
            settings["exit_slit"],
        )
        low_hv_res = energy_resolution_from_beamline_slit(
            resolution_table_selected,
            low_hv,
            slit_size,
        )
        high_hv_res = energy_resolution_from_beamline_slit(
            resolution_table_selected,
            high_hv,
            slit_size,
        )
        # interpolate between nearest values
        return low_hv_res + (high_hv_res - low_hv_res) * (settings["hv"] - low_hv) / (
            high_hv - low_hv
        ) * (1000 if meV else 1)

    raise NotImplementedError


def thermal_broadening_estimate(
    data: XrTypes,
    *,
    meV: bool = False,  # noqa: N803
) -> float:
    """Calculates the thermal broadening from the temperature on the data."""
    return normalize_to_spectrum(data).S.temp * K_BOLTZMANN_MEV_KELVIN * (1 if meV else 0.001)


def total_resolution_estimate(
    data: xr.DataArray,
    *,
    include_thermal_broadening: bool = False,
    meV: bool = False,  # noqa: N803
) -> float:
    """Gives the quadrature sum estimate of the resolution of an ARPES spectrum.

    For synchrotron ARPES, this typically means the scan has the photon energy,
    exit slit information and analyzer slit settings

    Returns:
        The estimated total resolution broadening.
    """
    thermal_broadening = 0.0
    if include_thermal_broadening:
        thermal_broadening = thermal_broadening_estimate(data, meV=meV)
    return math.sqrt(
        beamline_resolution_estimate(data, meV=meV) ** 2
        + analyzer_resolution_estimate(data, meV=meV) ** 2
        + thermal_broadening**2,
    )
