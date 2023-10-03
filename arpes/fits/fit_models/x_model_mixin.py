"""Extends lmfit to support curve fitting on xarray instances."""
from __future__ import annotations

import operator
import warnings
from logging import DEBUG, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import GaussianModel

if TYPE_CHECKING:
    from typing import Any

    from _typeshed import Incomplete
    from lmfit.model import ModelResult
    from numpy.typing import NDArray

__all__ = ["XModelMixin", "gaussian_convolve"]


LOGLEVEL = DEBUG
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def dict_to_parameters(dict_of_parameters: dict[str, Any]) -> lf.Parameters:
    """[TODO:summary].

    Args:
        dict_of_parameters(dict[str, Any]): pass to lf.Parameters

        cf.) the parameter of lf.Parameter class
           name: str (name of parameter)
           value: float (value of parameter, optional)
           vary: bool (whether the parameter is varied during fit)
           min and max: float (boundary for the value)
           expr: str (Mathematical expression used to constrain the value)
           brute_step: float (step size for grid points in the brute method
           user_data: optional, User-definedable extra attribute


    Returns:
        lf.Parameters
    """
    params = lf.Parameters()

    for param_name, param in dict_of_parameters.items():
        params[param_name] = lf.Parameter(param_name, **param)

    return params


class XModelMixin(lf.Model):
    """A mixin providing curve fitting for ``xarray.DataArray`` instances.

    This amounts mostly to making `lmfit` coordinate aware, and providing
    a translation layer between xarray and raw np.ndarray instances.

    Subclassing this mixin as well as an lmfit Model class should bootstrap
    an lmfit Model to one that works transparently on xarray data.

    Alternatively, you can use this as a model base in order to build new models.

    The core method here is `guess_fit` which is a convenient utility that performs both
    a `lmfit.Model.guess`, if available, before populating parameters and
    performing a curve fit.

    __add__ and __mul__ are also implemented, to ensure that the composite model
    remains an instance of a subclass of this mixin.
    """

    n_dims = 1
    dimension_order = None

    def guess_fit(
        self,
        data: xr.DataArray,
        params: lf.Parameters | dict[str, Any] | None = None,
        weights: Incomplete | None = None,
        *,
        guess: bool = True,
        debug: bool = False,
        prefix_params: bool = True,
        transpose: bool = False,
        **kwargs: Incomplete,
    ) -> ModelResult:
        """Performs a fit on xarray data after guessing parameters.

        Params allows you to pass in hints as to what the values and bounds on parameters
        should be. Look at the lmfit docs to get hints about structure

        Args:
            data (xr.DataArray): [TODO:description]
            params (lf.Parameters|dict| None): Fitting parameters
            weights ([TODO:type]): [TODO:description]
            guess (bool): [TODO:description]
            debug (bool): [TODO:description]
            prefix_params: [TODO:description]
            transpose: [TODO:description]
            kwargs([TODO:type]): pass to lf.Model.guess (parent class)
                Additional keyword arguments, passed to model function.
        """
        if params is not None and not isinstance(params, lf.Parameters):
            params = dict_to_parameters(params)
        assert isinstance(params, lf.Parameters)
        coord_values = {}
        if "x" in kwargs:
            coord_values["x"] = kwargs.pop("x")

        real_data, flat_data = data, data

        new_dim_order = None
        if isinstance(data, xr.DataArray):
            real_data, flat_data = data.values, data.values
            assert len(real_data.shape) == self.n_dims

            if self.n_dims == 1:
                coord_values["x"] = data.coords[next(iter(data.indexes))].values
            else:

                def find_appropriate_dimension(dim_or_dim_list: str | list[str]) -> str:
                    if isinstance(dim_or_dim_list, str):
                        assert dim_or_dim_list in data.dims
                        return dim_or_dim_list
                    intersect = set(dim_or_dim_list).intersection(data.dims)
                    assert len(intersect) == 1
                    return next(iter(intersect))

                # resolve multidimensional parameters
                if self.dimension_order is None or all(d is None for d in self.dimension_order):
                    new_dim_order = data.dims
                else:
                    new_dim_order = [
                        find_appropriate_dimension(dim_options)
                        for dim_options in self.dimension_order
                    ]

                if list(new_dim_order) != list(data.dims):
                    warnings.warn("Transposing data for multidimensional fit.", stacklevel=2)
                    data = data.transpose(*new_dim_order)

                coord_values = {k: v.values for k, v in data.coords.items() if k in new_dim_order}
                real_data, flat_data = data.values, data.values.ravel()

        real_weights = weights
        if isinstance(weights, xr.DataArray) and isinstance(real_weights, xr.DataArray):
            if self.n_dims == 1:
                real_weights = real_weights.values
            elif new_dim_order is not None:
                real_weights = weights.transpose(*new_dim_order).values.ravel()
            else:
                real_weights = weights.values.ravel()

        if transpose:
            assert (
                len(data.dims) == 1
            ), "You cannot transpose (invert) a multidimensional array (scalar field)."
            cached_coordinate = next(iter(coord_values.values()))
            coord_values[next(iter(coord_values.keys()))] = real_data
            real_data = cached_coordinate
            flat_data = real_data

        guessed_params: lf.Parameters = (
            self.guess(real_data, **coord_values) if guess else self.make_params()
        )

        if params is not None:
            for k, v in params.items():
                if isinstance(v, dict):
                    if prefix_params:
                        guessed_params[self.prefix + k].set(**v)
                    else:
                        guessed_params[k].set(**v)

            guessed_params.update(params)

        result = None
        try:
            result = super().fit(
                flat_data,  # Array of data to be fit
                guessed_params,  # lf.Parameters
                **coord_values,
                weights=real_weights,  # weights to use for the calculation of the fit residual
                **kwargs,
            )
            result.independent = coord_values
            result.independent_order = new_dim_order
        except Exception as e:
            print(e)
            if debug:
                import pdb

                pdb.post_mortem(e.__traceback__)
        finally:
            return result

    def xguess(self, data: xr.DataArray, **kwargs: Incomplete) -> lf.Parameters:
        """Tries to determine a guess for the parameters."""
        x = kwargs.pop("x", None)

        real_data = data
        if isinstance(data, xr.DataArray):
            real_data = data.values
            assert len(real_data.shape) == 1
            x = data.coords[next(iter(data.indexes))].values

        return self.guess(real_data, x=x, **kwargs)

    def __add__(self, other: XModelMixin) -> lf.Model:
        """Implements `+`."""
        comp = XAdditiveCompositeModel(self, other, operator.add)
        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp

    def __mul__(self, other: XModelMixin) -> lf.Model:
        """Implements `*`."""
        comp = XMultiplicativeCompositeModel(self, other, operator.mul)

        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp


class XAdditiveCompositeModel(lf.CompositeModel, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models."""

    def guess(
        self,
        data: xr.DataArray | xr.Dataset,
        x: NDArray[np.float_] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XMultiplicativeCompositeModel(lf.CompositeModel, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models.

    Currently this just copies ``+``, might want to adjust things!
    """

    def guess(
        self,
        data: xr.DataArray | xr.Dataset,
        x: NDArray[np.float_] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XConvolutionCompositeModel(lf.CompositeModel, XModelMixin):
    """Work in progress for convolving two ``Model``."""

    def guess(
        self,
        data: xr.DataArray | xr.Dataset,
        x: NDArray[np.float_] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}

        for c in self.components:
            if c.prefix == "conv_":
                # don't guess on the convolution term
                continue

            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


def gaussian_convolve(model_instance: Incomplete) -> lf.Model:
    """Produces a model that consists of convolution with a Gaussian kernel."""
    return XConvolutionCompositeModel(model_instance, GaussianModel(prefix="conv_"), np.convolve)
