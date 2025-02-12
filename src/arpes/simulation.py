"""Phenomenological models and detector simulation for accurate modelling of ARPES data.

Currently we offer relatively rudimentary detecotor modeling, mostly providing
only nonlinearity and some stubs, but a future release will provide reasonably
accurate modeling of the trapezoidal effect in hemispherical analyzers,
fixed mode artifacts, dust, and more.

Additionally we offer the ability to model the detector response at the
level of individual electron events using a point spread or more complicated
response. This allows the creation of reasonably realistic spectra for testing
new analysis techniques or working on machine learning based approaches that must
be robust to the shortcomings of actual ARPES data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
import scipy.signal as sig
import xarray as xr
from numpy.random import default_rng

from .constants import K_BOLTZMANN_MEV_KELVIN

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

__all__ = (
    # Composable detector effects, to simulate the real response of
    # an ARPES detector
    "DetectorEffect",
    "DustDetectorEffect",
    "FixedModeDetectorEffect",
    # Particular Detector Models
    "NonlinearDetectorEffect",
    # Spectral Function representation
    "SpectralFunction",
    # implementations of particular spectral functions
    "SpectralFunctionBSSCO",
    "SpectralFunctionMFL",
    "SpectralFunctionPhaseCoherent",
    "TrapezoidalDetectorEffect",
    "WindowedDetectorEffect",
    "apply_psf_to_point_cloud",
    # sampling utilities
    "cloud_to_arr",
    "sample_from_distribution",
)


class DetectorEffect:
    """Detector effects are callables that map a spectrum into a new transformed one.

    This might be used to imprint the image of a grid,
    dust, or impose detector nonlinearities.
    """

    def __call__(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Applies the detector effect to a spectrum.

        By default, apply the identity function.

        Args:
            spectrum: The input spectrum before modification.

        Returns:
            The data with this effect applied.
        """
        return spectrum


@dataclass
class NonlinearDetectorEffect(DetectorEffect):
    """Implements power law detector nonlinearities."""

    gamma: float | None = 1.0

    def __call__(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Applies the detector effect to a spectrum.

        The effect is modeled by letting the output intensity (I_out) be equal to
        the input intensity (I_in) to a fixed power. I.e. I_out[i, j] = I_in[i, j]^gamma.

        Args:
            spectrum: The input spectrum before modification.

        Returns:
            The data with this effect applied.
        """
        if self.gamma is not None:
            return spectrum**self.gamma

        msg = "Nonlinearity lookup tables are not yet supported."
        raise NotImplementedError(msg)


@dataclass
class FixedModeDetectorEffect(DetectorEffect):
    """Implements a grid or pore structure of an MCP or field termination mesh.

    Talk to Danny or Sam about getting hyperuniform point cloud distributions to use
    for the pore structure. Otherwise, we can just use a sine-grid.

    Attributes:
        spacing: The pixel periodicity of the pores.
        periodic: The grid type to use for the pores. One of ["hex"].
    """

    spacing: float = 5.0
    periodic: str = "hex"

    _cached_pore_structure = None

    @property
    def detector_imprint(self) -> xr.DataArray:
        """Provides the transmission factor for the grid on the spectrometer or "imprint"."""
        raise NotImplementedError

    def __call__(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Applies the detector effect to a spectrum.

        Args:
            spectrum: The input spectrum before modification.

        Returns:
            The data with this effect applied.
        """
        # will fail if we do not have the right size
        return self.detector_imprint * spectrum


class DustDetectorEffect(DetectorEffect):
    """Applies aberrations in the spectrum coming from dust.

    TODO, dust.
    """


class TrapezoidalDetectorEffect(DetectorEffect):
    """Applies trapezoidal detector windowing.

    TODO model that phi(pixel) is also a function of binding energy,
    i.e. that the detector has severe aberrations at low photoelectron
    kinetic energy (high retardation ratio).
    """


class WindowedDetectorEffect(DetectorEffect):
    """TODO model the finite width of the detector window as recorded on a camera."""


def cloud_to_arr(
    point_cloud: list[list[float]] | Iterable[NDArray[np.float64]],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Converts a point cloud (list of xy pairs) to an array representation.

    Uses linear interpolation for points that have non-integral coordinates.

    Args:
        point_cloud: The sampled set of electrons.
        shape: The shape of the desired output array.

    Returns:
        An array with the electron arrivals smeared into it.
    """
    cloud_as_image = np.zeros(shape)

    for x, y in zip(*point_cloud, strict=True):
        frac_low_x = 1 - (x - np.floor(x))
        frac_low_y = 1 - (y - np.floor(y))
        shape_x, shape_y = shape
        cloud_as_image[int(np.floor(x)) % shape_x][int(np.floor(y)) % shape_y] += (
            frac_low_x * frac_low_y
        )
        cloud_as_image[(int(np.floor(x)) + 1) % shape_x][int(np.floor(y)) % shape_y] += (
            1 - frac_low_x
        ) * frac_low_y
        cloud_as_image[int(np.floor(x)) % shape_x][(int(np.floor(y)) + 1) % shape_y] += (
            frac_low_x * (1 - frac_low_y)
        )
        cloud_as_image[(int(np.floor(x)) + 1) % shape_x][(int(np.floor(y)) + 1) % shape_y] += (
            1 - frac_low_x
        ) * (1 - frac_low_y)

    return cloud_as_image


def apply_psf_to_point_cloud(
    point_cloud: list[list[float]] | Iterable[NDArray[np.float64]],
    shape: tuple[int, int],
    sigma: tuple[int, int] = (10, 3),  # Note: Pixel units
) -> NDArray[np.float64]:
    """Takes a point cloud and turns it into a broadened spectrum.

    Samples are drawn individually and smeared by a
    gaussian PSF (Point spread function) given through the `sigma` parameter. Their net contribution
    as an integrated image is returned.

    In the future, we should also allow for specifying a particular PSF.

    Args:
        point_cloud: The sampled set of electrons.
        shape: The shape of the desired output array.
        sigma: The broadening to apply, in pixel units.

    Returns:
        An array with the electron arrivals smeared into it.
    """
    as_img = cloud_to_arr(point_cloud, shape)

    return scipy.ndimage.gaussian_filter(
        as_img,
        sigma=sigma,
        order=0,
        mode="reflect",
    )


def sample_from_distribution(
    distribution: xr.DataArray,
    n: int = 5000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Samples events from a probability distribution.

    Given a probability distribution in ND modeled by an array providing the PDF,
    sample individual events coming from this PDF.

    Args:
        distribution(xr.DataArray): The probability density. The probability of drawing a
                                    sample at (i, j) will be proportional to `distribution[i, j]`.
        n(int): The desired number of electrons/samples to pull from the distribution.

    Returns:
        An array with the arrival locations.
    """
    cdf_rows = np.cumsum(np.sum(distribution.values, axis=1))
    norm_rows = np.cumsum(
        distribution.values
        / np.expand_dims(
            np.sum(
                distribution.values,
                axis=1,
            ),
            axis=1,
        ),
        axis=1,
    )

    total = np.sum(distribution.values)

    rg = default_rng()
    sample_xs = np.searchsorted(
        cdf_rows,
        rg.random(n) * total,
    )
    sample_ys_rows = norm_rows[sample_xs, :]

    # take N samples between 0 and 1, which is now the normalized full range of the data
    # and find the index, this effectively samples the index in the array if it were a PDF
    sample_ys = []
    random_ys = rg.random(n)
    for random_y, row_y in zip(random_ys, sample_ys_rows, strict=True):
        sample_ys.append(np.searchsorted(row_y, random_y))
    return (np.asarray(sample_xs, float) + rg.random(n)), (
        np.asarray(np.array(sample_ys), float) + rg.random(n)
    )


class SpectralFunction:
    """Generic spectral function model for band with self energy in the single-particle picture."""

    def digest_to_json(self) -> dict[str, Any]:
        """Summarizes the parameters for the model to JSON."""
        return {"omega": self.omega, "temperature": self.temperature, "k": self.k}

    def fermi_dirac(self, omega: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculates the Fermi-Dirac occupation factor at energy values `omega`."""
        return 1 / (np.exp(omega / (K_BOLTZMANN_MEV_KELVIN * self.temperature)) + 1)

    def __init__(
        self,
        k: NDArray[np.float64] | None = None,
        omega: NDArray[np.float64] | None = None,
        temperature: float = 20,
    ) -> None:
        """Initialize from parameters.

        Args:
            k: The momentum range for the simulation.
            omega: The energy range for the simulation.
            temperature: The temperature for the simulation.
        """
        if k is None:
            k = np.linspace(-200, 200, 800, dtype=np.float64)
        elif len(k) == len(("start", "stop", "num")):
            k = np.linspace(*k, dtype=np.float64)

        if omega is None:
            omega = np.linspace(-1000, 1000, 2000, dtype=np.float64)
        elif len(omega) == len(("start", "stop", "num")):
            omega = np.linspace(*omega, dtype=np.float64)

        assert isinstance(k, np.ndarray)
        assert isinstance(omega, np.ndarray)

        self.temperature = temperature
        self.omega = omega
        self.k = k

    def imag_self_energy(self) -> NDArray[np.float64]:
        """Provides the imaginary part of the self energy."""
        return np.zeros(
            shape=self.omega.shape,
        )

    def real_self_energy(self) -> NDArray[np.complex128]:
        """Defaults to using Kramers-Kronig from the imaginary self energy."""
        return np.imag(sig.hilbert(self.imag_self_energy()))

    def self_energy(self) -> NDArray[np.complex128]:
        """Combines the self energy terms into a complex valued array."""
        return self.real_self_energy() + 1.0j * self.imag_self_energy()

    def bare_band(self) -> NDArray[np.float64]:
        """Provides the bare band dispersion."""
        return 3 * self.k

    def sampled_spectral_function(
        self,
        n_electrons: int = 50000,
        n_cycles: int = 1,
        psf_width: tuple[int, int] = (7, 3),
    ) -> xr.DataArray:
        """Samples electrons from the measured spectral function to calculate a detector image.

        The measured spectral function is used as a 2D density for the electrons. Samples are drawn
        and then broadened by a point spread (`psf`) modeling finite resolution detector response.

        Args:
            n_electrons: The number of electrons to draw.
            n_cycles: The number of frames to draw. `n_electrons` are drawn per cycle.
            psf_width: The point spread width in pixels.

        Returns:
            xr.DataArray: [description]
        """
        spectral = self.measured_spectral_function()
        sampled = [
            apply_psf_to_point_cloud(
                sample_from_distribution(spectral, n=n_electrons),
                (spectral.values.shape[0], spectral.values.shape[1]),
                sigma=psf_width,
            )
            for _ in range(n_cycles)
        ]

        new_coords = dict(spectral.coords)
        new_coords["cycle"] = np.array(range(n_cycles))
        return xr.DataArray(
            np.stack(sampled, axis=-1),
            coords=new_coords,
            dims=[*list(spectral.dims), "cycle"],
        )

    def measured_spectral_function(self) -> xr.DataArray:
        """Calculates the measured spectral function under practical conditions."""
        return self.occupied_spectral_function()

    def occupied_spectral_function(self) -> xr.DataArray:
        """Calculates the spectral function weighted by the thermal occupation."""
        spectral = self.spectral_function()
        spectral.values = spectral.values * np.expand_dims(self.fermi_dirac(self.omega), axis=1)
        return spectral

    def spectral_function(self) -> xr.DataArray:
        """Calculates spectral function according to the self energy modification of the bare band.

        This essentially implements the classic formula for the single particle spectral function as
        the Lorentzian broadened and offset bare band.

        Returns:
            An `xr.DataArray` with the spectral function intensity in a given momentum-energy window
        """
        self_energy = self.self_energy()
        imag_self_energy = np.imag(self_energy)
        real_self_energy = np.real(self_energy)

        bare = self.bare_band()
        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        numerator = np.outer(np.abs(imag_self_energy), np.ones(shape=bare.shape))
        data = numerator / (
            (full_omegas - np.expand_dims(bare, axis=0) - np.expand_dims(real_self_energy, axis=1))
            ** 2
            + np.expand_dims(imag_self_energy**2, axis=1)
        )
        return xr.DataArray(data, coords={"k": self.k, "omega": self.omega}, dims=["omega", "k"])


class SpectralFunctionMFL(SpectralFunction):  # pylint: disable=invalid-name
    """Implements the Marginal Fermi Liquid spectral function, more or less."""

    def digest_to_json(self) -> dict[str, Any]:
        """Summarizes the parameters for the model to JSON."""
        return {
            **super().digest_to_json(),
            "a": self.a,
            "b": self.a,
        }

    def __init__(
        self,
        k: NDArray[np.float64] | None = None,
        omega: NDArray[np.float64] | None = None,
        temperature: float = 20,
        mfl_parameter: tuple[float, float] = (10.0, 1.0),
    ) -> None:
        """Initializes from parameters.

        Args:
            k: The momentum axis.
            omega: The energy axis.
            temperature: The temperature to use for the calculation. Defaults to None.
            mfl_parameter (tuple[float, float]): The MFL parameter ('a', and 'b').
              Defaults to (10.0, 1.0)
        """
        super().__init__(k, omega, temperature)

        self.a, self.b = mfl_parameter

    def imag_self_energy(self) -> NDArray[np.float64]:
        """Calculates the imaginary part of the self energy."""
        return np.sqrt((self.a + self.b * self.omega) ** 2 + self.temperature**2)


class SpectralFunctionBSSCO(SpectralFunction):
    """Implements the spectral function for BSSCO as reported in PhysRevB.57.R11093.

    This spectral function is explored in the paper
    `"Collapse of superconductivity in cuprates via ultrafast quenching of phase coherence" <https://arxiv.org/pdf/1707.02305.pdf>`_.
    """

    def __init__(
        self,
        k: NDArray[np.float64] | None = None,
        omega: NDArray[np.float64] | None = None,
        temperature: float = 20,
        gap_parameters: tuple[float, float, float] = (50, 30, 0),
    ) -> None:
        """Initializes from parameters.

        Args:
            k: The momentum axis.
            omega: The energy axis.
            temperature: The temperature to use for the calculation. Defaults to None.
            delta: The gap size.
            gap_parameters (tuple[float, float, float]): Gap parameter of the BSSCO,
              Delta, and two Gamma pamaramters  (s- and p-wave)
        """
        self.delta, self.gamma_s, self.gamma_p = gap_parameters
        super().__init__(k, omega, temperature)

    def digest_to_json(self) -> dict[str, Any]:
        """Summarizes the parameters for the model to JSON."""
        return {
            **super().digest_to_json(),
            "delta": self.delta,
            "gamma_s": self.gamma_s,
            "gamma_p": self.gamma_p,
        }

    def self_energy(self) -> NDArray[np.complex128]:
        """Calculates the self energy."""
        shape = (len(self.omega), len(self.k))

        g_one = -1.0j * self.gamma_s * np.ones(shape=shape)
        bare = self.bare_band()

        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        return g_one + (self.delta**2) / (full_omegas + bare + 1.0j * self.gamma_p)

    def spectral_function(self) -> xr.DataArray:
        """Calculates spectral function according to the self energy modification of the bare band.

        This essentially implements the classic formula for the single particle spectral function as
        the Lorentzian broadened and offset bare band.

        Returns:
            An `xr.DataArray` of the spectral function intensity in a given momentum-energy window.
        """
        self_energy = self.self_energy()
        imag_self_energy = np.imag(self_energy)
        real_self_energy = np.real(self_energy)

        bare = self.bare_band()
        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        numerator = np.abs(imag_self_energy)
        data = numerator / (
            (full_omegas - np.expand_dims(bare, axis=0) - real_self_energy) ** 2
            + imag_self_energy**2
        )
        return xr.DataArray(data, coords={"k": self.k, "omega": self.omega}, dims=["omega", "k"])


class SpectralFunctionPhaseCoherent(SpectralFunctionBSSCO):
    """Implements the "phase coherence" model for the BSSCO spectral function."""

    def self_energy(self) -> NDArray[np.complex128]:
        """Calculates the self energy using the phase coherent BSSCO model."""
        shape = (len(self.omega), len(self.k))

        g_one = (
            -1.0j
            * self.gamma_s
            * np.ones(shape=shape)
            * np.sqrt(1 + 0.0005 * np.expand_dims(self.omega, axis=1) ** 2)
        )
        bare = self.bare_band()

        full_omegas = np.outer(self.omega, np.ones(shape=bare.shape))

        self_e = g_one + (self.delta**2) / (full_omegas + bare + 1.0j * self.gamma_p)
        imag_self_e = np.imag(self_e)
        np.imag(sig.hilbert(imag_self_e, axis=0))
        rg = default_rng()

        return self_e + 3 * (rg.random(self_e.shape) + rg.random(self_e.shape) * 1.0j)
