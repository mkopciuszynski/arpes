"""Definitions of band shape.  These models are supporsed to be used for after broadcast_model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


__all__ = (" ParabolicDispersionPhiModel",)


class ParabolicDispersionPhiModel(XModelMixin):
    """Model for Parabolic Band model for ARPES data (angle-energy relationshipo)."""

    def parabolic_band_dispersion_phi(
        self,
        x: NDArray[np.float64],
        effective_mass: float,
        phi_offset: float,
        energy_offset: float,
    ) -> NDArray[np.float64]:
        """Return the energy at the emission angle under the free electron band model."""
        return


def parabolic_band_dispersion_angle(
    theta_degree: A,
    e0: float,
    mass: float = 1.0,
) -> A:
    """Return the energy at the given angle of emission (Free electron band).

    Energy reference is the vacuum level.
    (i.e. the energy is the kinetic energy, not final state energy)


    Parameters
    ----------
    theta_degree : float
        emission angle
    e0 : float
        energy at the Gamma point
    mass : float, optional
        electron mass, the static electron unit, by default 1.0

    Returns:
    -------
    float
        Energy in eV unit measured from the vacuum level.

    """
    assert isinstance(theta_degree, np.ndarray | float)
    return e0 * mass / (mass - np.sin(np.deg2rad(theta_degree)) ** 2)
