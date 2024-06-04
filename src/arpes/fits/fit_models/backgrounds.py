"""Definitions of common backgrounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import numpy as np
from lmfit.models import update_param_vals

from .functional_forms import affine_bkg
from .x_model_mixin import XModelMixin

if TYPE_CHECKING:
    import lmfit as lf
    import xarray as xr
    from numpy.typing import NDArray

    from arpes.fits import ModelArgs

__all__ = ("AffineBackgroundModel",)


class AffineBackgroundModel(XModelMixin):
    """A model for an affine (linear) background."""

    def __init__(self, **kwargs: Unpack[ModelArgs]) -> None:
        """Defer to lmfit for initialization."""
        kwargs.setdefault("prefix", "")
        kwargs.setdefault("independent_vars", ["x"])
        kwargs.setdefault("nan_policy", "raise")
        super().__init__(affine_bkg, **kwargs)

    def guess(
        self,
        data: xr.DataArray | NDArray[np.float_],
        x: NDArray[np.float_],
        **kwargs: float,
    ) -> lf.Parameters:
        """Use the tenth percentile value for the slope and a zero offset.

        Generally this should converge well regardless.
        """
        del x
        pars = self.make_params()

        pars[f"{self.prefix}lin_bkg"].set(value=np.percentile(data, 10))
        pars[f"{self.prefix}const_bkg"].set(value=0)

        return update_param_vals(pars, self.prefix, **kwargs)
