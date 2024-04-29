"""Utilities related to curve-fitting of ARPES data and xarray format data."""

from __future__ import annotations

from typing import Literal, TypedDict

from .fit_models.backgrounds import AffineBackgroundModel
from .fit_models.decay import ExponentialDecayCModel, TwoExponentialDecayCModel
from .fit_models.dirac import DiracDispersionModel
from .fit_models.fermi_edge import (
    AffineBroadenedFD,
    BandEdgeBGModel,
    BandEdgeBModel,
    FermiDiracAffGaussModel,
    FermiDiracModel,
    FermiLorentzianModel,
    GStepBModel,
    GStepBStandardModel,
    GStepBStdevModel,
    TwoBandEdgeBModel,
    TwoLorEdgeModel,
)
from .fit_models.misc import (
    FermiVelocityRenormalizationModel,
    LogRenormalizationModel,
    QuadraticModel,
)
from .fit_models.two_dimensional import EffectiveMassModel, Gaussian2DModel
from .fit_models.wrapped import (
    ConstantModel,
    GaussianModel,
    LinearModel,
    LogisticModel,
    LorentzianModel,
    SineModel,
    SkewedVoigtModel,
    SplitLorentzianModel,
    StepModel,
    VoigtModel,
)
from .fit_models.x_model_mixin import XModelMixin, gaussian_convolve
from .utilities import broadcast_model, result_to_hints

NAN_POLICY = Literal["raise", "propagate", "omit"]


class ModelArgs(TypedDict, total=False):
    """KWargs for lf.Model."""

    independent_vars: list[str]
    param_names: list[str]
    nan_policy: NAN_POLICY
    prefix: str
    name: str
    form: str
