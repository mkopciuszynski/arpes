"""Implements 2D and 3D angle scan momentum conversion for Fermi surfaces.

Broadly, this covers cases where we are not performing photon energy scans.
"""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import numba
import numpy as np

from arpes.constants import K_INV_ANGSTROM

from .base import K_SPACE_BORDER, MOMENTUM_BREAKPOINTS, CoordinateConverter
from .bounds_calculations import calculate_kp_bounds, calculate_kx_ky_bounds
from .calibration import DetectorCalibration

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import MOMENTUM

__all__ = ["ConvertKp", "ConvertKxKy"]


LOGLEVEL = (DEBUG, INFO)[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@numba.njit(parallel=True)
def _exact_arcsin(  # noqa: PLR0913
    k_par: NDArray[np.float_],
    k_perp: NDArray[np.float_],
    k_tot: NDArray[np.float_],
    phi: NDArray[np.float_],
    offset: float,
    *,
    par_tot: bool,
    negate: bool,
) -> None:
    """A efficient arcsin with total momentum scaling."""
    mul_idx = 1 if par_tot else 0
    for i in numba.prange(len(k_par)):
        result = np.arcsin(k_par[i] / np.sqrt(k_tot[i * mul_idx] ** 2 - k_perp[i] ** 2))
        if negate:
            result = -result
        phi[i] = result + offset


@numba.njit(parallel=True)
def _small_angle_arcsin(  # noqa: PLR0913
    k_par: NDArray[np.float_],
    k_tot: NDArray[np.float_],
    phi: NDArray[np.float_],
    offset: float,
    *,
    par_tot: bool,
    negate: bool,
) -> None:
    """A efficient small angle arcsin with total momentum scaling.

    np.arcsin(k_par / k_tot, phi)
    phi += offset
    mul_idx = 0
    """
    mul_idx = 1 if par_tot else 0
    for i in numba.prange(len(k_par)):
        result = np.arcsin(k_par[i] / k_tot[i * mul_idx])
        if negate:
            result = -result
        phi[i] = result + offset


@numba.njit(parallel=True)
def _rotate_kx_ky(
    kx: NDArray[np.float_],
    ky: NDArray[np.float_],
    kxout: NDArray[np.float_],
    kyout: NDArray[np.float_],
    chi: float,
) -> None:
    cos_chi = np.cos(chi)
    sin_chi = np.sin(chi)
    for i in numba.prange(len(kx)):
        kxout[i] = kx[i] * cos_chi - ky[i] * sin_chi
        kyout[i] = ky[i] * cos_chi + kx[i] * sin_chi


@numba.njit(parallel=True)
def _compute_ktot(
    hv: float,
    work_function: float,
    binding_energy: NDArray[np.float_],
    k_tot: NDArray[np.float_],
) -> None:
    for i in numba.prange(len(binding_energy)):
        k_tot[i] = K_INV_ANGSTROM * np.sqrt(hv - work_function + binding_energy[i])


def _safe_compute_k_tot(
    hv: float,
    work_function: float,
    binding_energy: float | NDArray[np.float_] | xr.DataArray,
) -> NDArray[np.float_]:
    arr_binding_energy = binding_energy
    if not isinstance(binding_energy, np.ndarray):
        arr_binding_energy = np.array([binding_energy])
    k_tot = np.zeros_like(arr_binding_energy)
    _compute_ktot(hv, work_function, arr_binding_energy, k_tot)
    return k_tot


class ConvertKp(CoordinateConverter):
    """A momentum converter for single ARPES (kp) cuts."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Initialize cached coordinates.

        Args:
            args: Pass to CoordinateConverter
            kwargs: Pass to CoordinateConverter

        Memo: Arguments of CoordinateConverter
            arr: xr.DataArray,
            dim_order: list[str] | None = None,
            calibration: DetectorCalibration | None = None,
        """
        super().__init__(*args, **kwargs)
        self.k_tot: NDArray[np.float_] | None = None
        self.phi: NDArray[np.float_] | None = None

    def get_coordinates(
        self,
        resolution: dict[MOMENTUM, float] | None = None,
        bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    ) -> dict[str, NDArray[np.float_]]:
        """Calculates appropriate coordinate bounds.

        Args:
            resolution(dict[MOMENTUM, float]): Represents conversion resolution
                key: momentum name, such as "kp", value: resolution, typical value is 0.001
            bounds(dict[str, Iterable[float]], optional): the key is the axis name.
                                                          the value represents the bounds

        Returns:
            dict[str, NDArray]: the key represents the axis name suchas "kp", "kx", and "eV".
        """
        if resolution is None:
            resolution = {}
        if bounds is None:
            bounds = {}
        coordinates = super().get_coordinates(resolution, bounds=bounds)
        (kp_low, kp_high) = calculate_kp_bounds(self.arr)
        if "kp" in bounds:
            kp_low, kp_high = bounds["kp"]
        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["phi"])
        try:
            inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][
                -2 if (len(self.arr.coords["phi"]) < 80) else -1  # noqa: PLR2004
            ]
        except IndexError:
            inferred_kp_res = MOMENTUM_BREAKPOINTS[-2]
        coordinates["kp"] = np.arange(
            kp_low - K_SPACE_BORDER,
            kp_high + K_SPACE_BORDER,
            resolution.get("kp", inferred_kp_res),
        )
        base_coords = {
            str(k): v for k, v in self.arr.coords.items() if k not in ["eV", "phi", "beta", "theta"]
        }  # should v.values ?
        coordinates.update(base_coords)
        return coordinates

    def compute_k_tot(self, binding_energy: NDArray[np.float_]) -> None:
        """Compute the total momentum (inclusive of kz) at different binding energies."""
        if self.arr.S.energy_notation == "Binding":
            self.k_tot = _safe_compute_k_tot(
                self.arr.S.hv,
                self.arr.S.analyzer_work_function,
                binding_energy,
            )
        elif self.arr.S.energy_notation == "Kinetic":
            self.k_tot = _safe_compute_k_tot(0, self.arr.S.analyzer_work_function, binding_energy)
        else:
            warning_msg = "Energy notation is not specified. Assume the Binding energy notation"
            warnings.warn(
                warning_msg,
                stacklevel=2,
            )
            self.k_tot = _safe_compute_k_tot(
                self.arr.S.hv,
                self.arr.S.analyzer_work_function,
                binding_energy,
            )

    def kspace_to_phi(
        self,
        binding_energy: NDArray[np.float_],
        kp: NDArray[np.float_],
        *args: Incomplete,
    ) -> NDArray[np.float_]:
        """Converts from momentum back to the analyzer angular axis."""
        # Dont remove *args even if not used.
        del args
        if self.phi is not None:
            return self.phi
        if self.is_slit_vertical:
            polar_angle = self.arr.S.lookup_offset_coord("theta") + self.arr.S.lookup_offset_coord(
                "psi",
            )
            parallel_angle = self.arr.S.lookup_offset_coord("beta")
        else:
            polar_angle = self.arr.S.lookup_offset_coord("beta") + self.arr.S.lookup_offset_coord(
                "psi",
            )
            parallel_angle = self.arr.S.lookup_offset_coord("theta")
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)
        self.phi = np.zeros_like(kp)
        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert self.k_tot is not None
        assert len(self.k_tot) == len(kp) or len(self.k_tot) == 1
        _small_angle_arcsin(
            kp / np.cos(polar_angle),
            self.k_tot,
            self.phi,
            self.arr.S.phi_offset + parallel_angle,
            par_tot=par_tot,
            negate=False,
        )
        if isinstance(self.calibration, DetectorCalibration):
            self.phi = self.calibration.correct_detector_angle(eV=binding_energy, phi=self.phi)
        assert self.phi is not None
        return self.phi

    def conversion_for(self, dim: str) -> Callable[[NDArray[np.float_]], NDArray[np.float_]]:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def _with_identity(*args: NDArray[np.float_]) -> NDArray[np.float_]:
            return self.identity_transform(dim, *args)

        return {  # type: ignore[return-value]
            "eV": self.kspace_to_BE,
            "phi": self.kspace_to_phi,
        }.get(
            dim,
            _with_identity,
        )


class ConvertKxKy(CoordinateConverter):
    """Implements volumetric momentum conversion for kx-ky scans.

    Please note that currently we assume that psi = 0 when you are not using an
    electrostatic deflector.
    """

    def __init__(self, arr: xr.DataArray, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Initialize the kx-ky momentum converter and cached coordinate values."""
        super().__init__(arr, *args, **kwargs)
        self.k_tot: NDArray[np.float_] | None = None
        # the angle perpendicular to phi as appropriate to the scan, this can be any of
        # psi, theta, beta
        self.perp_angle: NDArray[np.float_] | None = None
        self.rkx: NDArray[np.float_] | None = None
        self.rky: NDArray[np.float_] | None = None
        # accept either vertical or horizontal, fail otherwise
        if not any(
            np.abs(arr.alpha - alpha_option) < np.deg2rad(1) for alpha_option in [0, np.pi / 2]
        ):
            msg = "You must convert either vertical or horizontal slit data with this converter."
            raise ValueError(
                msg,
            )
        self.direct_angles = ("phi", next(d for d in ["psi", "beta", "theta"] if d in arr.indexes))
        if self.direct_angles[1] != "psi":
            # psi allows for either orientation
            assert (self.direct_angles[1] in {"theta"}) != (not self.is_slit_vertical)
        # determine which other angles constitute equivalent sets
        opposite_direct_angle = "theta" if "psi" in self.direct_angles else "psi"
        if self.is_slit_vertical:
            self.parallel_angles = (
                "beta",
                opposite_direct_angle,
            )
        else:
            self.parallel_angles = (
                "theta",
                opposite_direct_angle,
            )

    def get_coordinates(
        self,
        resolution: dict[MOMENTUM, float] | None = None,
        bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    ) -> dict[str, NDArray[np.float_]]:
        """Calculates appropriate coordinate bounds."""
        if resolution is None:
            resolution = {}
        if bounds is None:
            bounds = {}
        coordinates = super().get_coordinates(resolution, bounds=bounds)
        ((kx_low, kx_high), (ky_low, ky_high)) = calculate_kx_ky_bounds(self.arr)
        if "kx" in bounds:
            kx_low, kx_high = bounds["kx"]
        if "ky" in bounds:
            ky_low, ky_high = bounds["ky"]
        kx_angle, ky_angle = self.direct_angles
        if self.is_slit_vertical:
            # phi actually measures along ky
            ky_angle, kx_angle = kx_angle, ky_angle
            ((ky_low, ky_high), (kx_low, kx_high)) = ((kx_low, kx_high), (ky_low, ky_high))
        len_ky_angle = len(self.arr.coords[ky_angle])
        len_kx_angle = len(self.arr.coords[kx_angle])
        inferred_kx_res = (kx_high - kx_low + 2 * K_SPACE_BORDER) / len(self.arr.coords[kx_angle])
        inferred_ky_res = (ky_high - ky_low + 2 * K_SPACE_BORDER) / len(self.arr.coords[ky_angle])
        # upsample a bit if there aren't that many points along a certain axis
        try:
            inferred_kx_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kx_res][
                -2 if (len_kx_angle < 80) else -1  # noqa: PLR2004
            ]
        except IndexError:
            inferred_kx_res = MOMENTUM_BREAKPOINTS[-2]
        try:
            inferred_ky_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_ky_res][
                -2 if (len_ky_angle < 80) else -1  # noqa: PLR2004
            ]
        except IndexError:
            inferred_ky_res = MOMENTUM_BREAKPOINTS[-2]
        coordinates["kx"] = np.arange(
            kx_low - K_SPACE_BORDER,
            kx_high + K_SPACE_BORDER,
            resolution.get("kx", inferred_kx_res),
        )
        coordinates["ky"] = np.arange(
            ky_low - K_SPACE_BORDER,
            ky_high + K_SPACE_BORDER,
            resolution.get("ky", inferred_ky_res),
        )
        base_coords = {
            str(k): v  # should v.values?
            for k, v in self.arr.coords.items()
            if k not in ["eV", "phi", "psi", "theta", "beta", "alpha", "chi"]
        }
        coordinates.update(base_coords)
        return coordinates

    def compute_k_tot(self, binding_energy: NDArray[np.float_]) -> None:
        """Compute the total momentum (inclusive of kz) at different binding energies."""
        if self.arr.energy_notation == "Binding":
            self.k_tot = _safe_compute_k_tot(
                self.arr.S.hv,
                self.arr.S.analyzer_work_function,
                binding_energy,
            )
        elif self.arr.energy_notation == "Kinetic":
            self.k_tot = _safe_compute_k_tot(0, self.arr.S.analyzer_work_function, binding_energy)
        else:
            warning_msg = "Energy notation is not specified. Assume the Binding energy notation"
            warnings.warn(
                warning_msg,
                stacklevel=2,
            )
            self.k_tot = _safe_compute_k_tot(
                self.arr.S.hv,
                self.arr.S.analyzer_work_function,
                binding_energy,
            )

    def conversion_for(self, dim: str) -> Callable[[NDArray[np.float_]], NDArray[np.float_]]:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def _with_identity(*args: NDArray[np.float_]) -> NDArray[np.float_]:
            return self.identity_transform(dim, *args)

        return {  # type: ignore[return-value]
            "eV": self.kspace_to_BE,
            "phi": self.kspace_to_phi,
            "theta": self.kspace_to_perp_angle,
            "psi": self.kspace_to_perp_angle,
            "beta": self.kspace_to_perp_angle,
        }.get(dim, _with_identity)

    @property
    def needs_rotation(self) -> bool:
        """Whether we need to rotate the momentum coordinates when converting to angle."""
        # force rotation when greater than 0.5 deg
        return np.abs(self.arr.S.lookup_offset_coord("chi")) > np.deg2rad(0.5)

    def rkx_rky(
        self,
        kx: NDArray[np.float_],
        ky: NDArray[np.float_],
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Returns the rotated kx and ky values when we are rotating by nonzero chi."""
        if self.rkx is not None and self.rky is not None:
            return self.rkx, self.rky
        chi = self.arr.S.lookup_offset_coord("chi")
        self.rkx = np.zeros_like(kx)
        self.rky = np.zeros_like(ky)
        _rotate_kx_ky(kx, ky, self.rkx, self.rky, chi)
        return self.rkx, self.rky

    def kspace_to_phi(
        self,
        binding_energy: NDArray[np.float_],
        kx: NDArray[np.float_],
        ky: NDArray[np.float_],
        *args: Incomplete,
    ) -> NDArray[np.float_]:
        """Converts from momentum back to the analyzer angular axis."""
        logger.debug("the following args are not used in kspace_to_phi")
        logger.debug(args)
        if self.phi is not None:
            return self.phi
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)
        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)
        # This can be condensed but it is actually better not to condense it:
        # In this format, we can very easily compare to the raw coordinate conversion functions that
        # come from Mathematica in order to adjust signs, etc.
        scan_angle = self.direct_angles[1]
        self.phi = np.zeros_like(ky)
        offset = self.arr.S.phi_offset + self.arr.S.lookup_offset_coord(self.parallel_angles[0])
        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert self.k_tot is not None
        assert len(self.k_tot) == len(self.phi) or len(self.k_tot) == 1
        if scan_angle == "psi":
            if self.is_slit_vertical:
                _exact_arcsin(ky, kx, self.k_tot, self.phi, offset, par_tot=par_tot, negate=False)
            else:
                _exact_arcsin(kx, ky, self.k_tot, self.phi, offset, par_tot=par_tot, negate=False)
        elif scan_angle == "beta":
            # vertical slit
            _small_angle_arcsin(kx, self.k_tot, self.phi, offset, par_tot=par_tot, negate=False)
        elif scan_angle == "theta":
            # vertical slit
            _small_angle_arcsin(ky, self.k_tot, self.phi, offset, par_tot=par_tot, negate=False)
        else:
            msg = f"No recognized scan angle found for {self.parallel_angles[1]}"
            raise ValueError(
                msg,
            )
        if isinstance(self.calibration, DetectorCalibration):
            self.phi = self.calibration.correct_detector_angle(eV=binding_energy, phi=self.phi)
        return self.phi

    def kspace_to_perp_angle(
        self,
        binding_energy: NDArray[np.float_],
        kx: NDArray[np.float_],
        ky: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """Converts from momentum back to the scan angle perpendicular to the analyzer."""
        if self.perp_angle is not None:
            return self.perp_angle
        if self.k_tot is None:
            self.compute_k_tot(binding_energy)
        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)
        scan_angle = self.direct_angles[1]
        self.perp_angle = np.zeros_like(kx)
        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert self.k_tot is not None
        assert len(self.k_tot) == len(self.perp_angle) or len(self.k_tot) == 1
        if scan_angle == "psi":
            if self.is_slit_vertical:
                offset = self.arr.S.psi_offset - self.arr.S.lookup_offset_coord(
                    self.parallel_angles[1],
                )
                _small_angle_arcsin(
                    kx,
                    self.k_tot,
                    self.perp_angle,
                    offset,
                    par_tot=par_tot,
                    negate=True,
                )
            else:
                offset = self.arr.S.psi_offset + self.arr.S.lookup_offset_coord(
                    self.parallel_angles[1],
                )
                _small_angle_arcsin(
                    ky,
                    self.k_tot,
                    self.perp_angle,
                    offset,
                    par_tot=par_tot,
                    negate=False,
                )
        elif scan_angle == "beta":
            offset = self.arr.S.beta_offset + self.arr.S.lookup_offset_coord(
                self.parallel_angles[1],
            )
            _exact_arcsin(ky, kx, self.k_tot, self.perp_angle, offset, par_tot=par_tot, negate=True)
        elif scan_angle == "theta":
            offset = self.arr.S.theta_offset - self.arr.S.lookup_offset_coord(
                self.parallel_angles[1],
            )
            _exact_arcsin(kx, ky, self.k_tot, self.perp_angle, offset, par_tot=par_tot, negate=True)
        else:
            msg = f"No recognized scan angle found for {self.parallel_angles[1]}"
            raise ValueError(
                msg,
            )
        return self.perp_angle