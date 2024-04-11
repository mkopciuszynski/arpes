"""Useful constants for experiments and conversions.

Much of this is collected from past students, especially Jeff's 'Cstes.ipf'.

Some of this will disappear in future updates, as we move away from magic constants towards
bundling necessary information on endstation classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._typing import Spectrometer

# eV, A reasonablish value if you aren't sure for the particular sample
WORK_FUNCTION = 4.3

METERS_PER_SECOND_PER_EV_ANGSTROM = (
    151927  # converts from eV * angstrom to meters/second velocity units
)
HBAR = 1.0545718176461565e-34
HBAR_PER_EV = 6.582119569509067e-16
# gives the energy lifetime relationship via tau = -hbar / np.imag(self_energy)


BARE_ELECTRON_MASS = 9.109383701e-31  # kg
HBAR_SQ_EV_PER_ELECTRON_MASS = 0.475600805657  # hbar^2 / m0 in eV^2 s^2 / kg
HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ = 7.619964  # (hbar^2) / (m0 * angstrom ^2) in eV

K_BOLTZMANN_EV_KELVIN = 8.617333262145178e-5  # in units of eV / Kelvin
K_BOLTZMANN_MEV_KELVIN = 1000 * K_BOLTZMANN_EV_KELVIN  # meV / Kelvin

HC = 1239.8419843320028  # in units of eV * nm

# Lanzara lab specific
STRAIGHT_TOF_LENGTH = 0.937206
SPIN_TOF_LENGTH = 1.1456
DLD_LENGTH = 1.1456  # This isn't correct but it should be a reasonable guess

K_INV_ANGSTROM = 0.5123167219534328
HV_CONVERSION = 3.814697265625

TWO_DIMENSION = 2

# TODO: these should be migrated into their appropriate loaders
#
#

SPECTROMETER_MC: Spectrometer = {
    "name": "MC",
    "rad_per_pixel": np.deg2rad(1 / 10),
    "type": "hemisphere",
    "is_slit_vertical": False,
}

SPECTROMETER_MC_OLD: Spectrometer = {
    "name": "MC_OLD",
    "type": "hemisphere",
    "rad_per_pixel": np.deg2rad(0.125),
    "is_slit_vertical": False,
}

SPECTROMETER_STRAIGHT_TOF: Spectrometer = {
    "name": "STRAIGHT_ToF",
    "length": STRAIGHT_TOF_LENGTH,
    "mstar": 1.0,
    "type": "tof",
    "dof": ["t"],
    "scan_dof": ["theta"],
}

SPECTROMETER_SPIN_TOF: Spectrometer = {
    "name": "SPIN_ToF",
    "length": SPIN_TOF_LENGTH,
    "mstar": 0.5,
    "type": "tof",
    "dof": ["time", "spin"],
    "scan_dof": ["theta", "beta"],
}

SPECTROMETER_DLD: Spectrometer = {
    "name": "DLD",
    "length": DLD_LENGTH,
    "type": "tof",
    "dof_type": {
        "timing": ["x_pixels", "t_pixels"],
        "spatial": ["x_pixels", "y_pixels"],
    },
    "scan_dof": ["theta"],
}

SPECTROMETER_BL4: Spectrometer = {
    "name": "BL4",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}

SPECTROMETER_BL7: Spectrometer = {
    "name": "BL7",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}

SPECTROMETER_ANTARES: Spectrometer = {
    "name": "ANTARES",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}

SPECTROMETER_KAINDL: Spectrometer = {
    "name": "Kaindl",
    "is_slit_vertical": True,
    "type": "hemisphere",
    "dof": ["theta", "sample_phi"],
}


SPECTROMETERS = {
    "ALG-MC": SPECTROMETER_MC,
    "ALG-SToF": SPECTROMETER_STRAIGHT_TOF,
    "BL403": SPECTROMETER_BL4,
    "Kaindl": SPECTROMETER_KAINDL,
    "BL7": SPECTROMETER_BL7,
    "ANTARES": SPECTROMETER_ANTARES,
    "SToF": SPECTROMETER_SPIN_TOF,
    "ToF": SPECTROMETER_STRAIGHT_TOF,
    "DLD": SPECTROMETER_DLD,
}
