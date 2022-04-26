from typing import Iterable, Optional

from numpy import typing as npt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

from matching._typing import sklearn_types as skt


class UnsupportedModel(Exception):
    """Raised when an unsupported model is provided"""


def _get_preds_from_fittable(model: skt._Fittable, X: npt.ArrayLike) -> np.ndarray:
    """Utility function to get the predictions out of an sklearn model.
    If the model is a `_Classifier` and predicts probabilities, we take `predict_proba` output, dropping the first column.
    If the model is a `_Regressor`, we take `predict` output

    Parameters
    ----------
    model : :class:`_Fittable`
        Fitted object which we want to use to predict over `X`
    X : ArrayLike
        Of features to get predictions for

    Returns
    -------
    np.ndarray
        - if `model` is a :class:`_Classifier`, predicted probabilities, with the first column removed
        - if `model` is a :class:`_Regressor`, predicted values
    """
    # TODO: Docstring
    if isinstance(model, skt._Classifier):
        return model.predict_proba(X)[:, 1:]
    elif isinstance(model, skt._Regressor):
        return model.predict(X)
    else:
        raise UnsupportedModel("Model must be of type `skt._Classifier` or `skt._Fittable`")


def propensity_score(
    X: npt.ArrayLike, z: npt.ArrayLike, model: Optional[skt._Fittable] = None, use_logit: bool = False
) -> np.ndarray:
    """Compute propensity scores for the data. Propensity scores are predicted treatment assignments (`z`) from `X`.
    This fitted model is then used to get predicted scores over the WHOLE dataset.

    Parameters
    ----------
    X : ArrayLike
        of features to predict `y` with
    z : ArrayLike
        of binary treatment assignments
    model : Optional[:class:`_Fittable`]
        Model to use in prediction. Can be a regressor (for continuous `z`) or a classifier (for discrete `z`)
        By default, Logistic Regression with no penalty is used
    use_logit : bool, default=False
        Whether or not to take the logit of the predicted values. Only pertinent when `model` is a classifier (and predicts probabilities)

    Returns
    -------
    ndarray
        of propensity scores, aka predicted treatments, aka z-hat
    """
    if model is None:
        model = LogisticRegression(penalty="none")

    X, z = np.asarray(X), np.asarray(z)

    # Fit a model to predict treatment from features
    model.fit(X, z)

    # Get predictions from the fitted model
    preds = _get_preds_from_fittable(model, X)
    return logit(preds) if use_logit else preds


def prognostic_score(
    X: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike, model: Optional[skt._Fittable] = None, use_logit: bool = False
) -> np.ndarray:
    """Compute prognostic scores for the data.
    Prognostic scores are fit by predicting outcomes (`y`) from `X`, using only the CONTROL dataset.
    This fitted model is then used to get predicted scores over the WHOLE dataset.

    Parameters
    ----------
    X : ArrayLike
        of features to predict `y` with
    y : ArrayLike
        of outcomes
    z : ArrayLike
        of binary treatment assignments
    model : Optional[_Fittable]
        Model to use in prediction. Can be a regressor (for continuous response) or a classifier (for discrete response)
        By default, Logistic Regression with no penalty is used
    use_logit : bool = False
        Whether or not to take the logit of the predicted values. Only pertinent when `model` is a classifier (and predicts probabilities)

    Returns
    -------
    ndarray
        of prognostic scores, aka predicted outcomes, aka y-hat
    """
    if model is None:
        model = LogisticRegression(penalty="none")

    X, y, z = np.asarray(X), np.asarray(y), np.asarray(z)

    # Prognostic scores are computed using only data from control group
    Xc, yc = X[z, :], y[z, :]
    model.fit(Xc, yc)

    # ... we then use this model to predict over _all_ X
    preds = _get_preds_from_fittable(model, X)
    return logit(preds) if use_logit else preds


def autocoarsen(X: npt.ArrayLike, n_bins: npt.ArrayLike = 5) -> np.ndarray:
    """Automatically coarsen all columns of `X` to a certain number of bins

    Parameters
    ----------
    X : ArrayLike
        input array to coarsen
    n_bins : ArrayLike
        either a single `int`, or an array of `int`s, specifying number of bins for each column

    Returns
    -------
    ndarray
        of coarsened `X`
    """
    Xarr = pd.get_dummies(X).values
    if not not isinstance(n_bins, Iterable) or np.asarray(n_bins).shape[0] == Xarr.shape[1]:
        raise ValueError("n_bins must be either a single value, or of length `np.asarray(X).shape[1]`")

    # TODO: Should this be K-Means??
    return KBinsDiscretizer(n_bins).fit_transform(X)

def autocoarsen_cv(X: npt.ArrayLike, min_k: int = 1, max_k: int = 10) -> np.ndarray:
    """Automatically coarsen all columns of `X` to a certain number of bins"""
    # TODO: Docstring
    # Xarr = pd.get_dummies(X).values
    # if not not isinstance(n_bins, Iterable) or np.asarray(n_bins).shape[0] == Xarr.shape[1]:
    #     raise ValueError("n_bins must be either a single value, or of length `np.asarray(X).shape[1]`")

    # return KBinsDiscretizer(n_bins).fit_transform(X)
    raise NotImplemented
