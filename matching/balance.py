import numpy as np
from numpy import typing as npt


def standardized_mean_difference(a: npt.ArrayLike, b: npt.ArrayLike):
    """Computes the standardized mean difference between a and b.

    .. math:: \frac{\overline{a} - \overline{b}}{\sqrt{\text{Var}(a) + \text{Var}(b)}}

    Parameters
    ----------
    a, b : npt.ArrayLike
        to compute the standardized mean difference between

    Returns
    -------
    float
        standardized mean difference
    """
    meana, stda = np.mean(a), np.std(a)
    meanb, stdb = np.mean(b), np.std(b)

    return (meana - meanb) / np.sqrt(stda**2 + stdb**2)


def ecdf(x: npt.ArrayLike):
    """Fits an eCDF to `x` and returns a callable to evaluate it at any point(s)

    Parameters
    ----------
    x : ArrayLike
        to fit the eCDF to

    Returns
    -------
    inner : Callable
        evaluates the eCDF at point `v`
    """
    x_sorted = np.sort(np.asarray(x))

    def fitted(v: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluates the eCDF of `x` at point `v`

        Parameters
        ----------
        v : ArrayLike
            point(s) to evaluate the eCDF of fitted `x` at

        Returns
        -------
        ArrayLike
            of eCDF evaluations in [0, 1]
        """
        return np.searchsorted(x_sorted, v, side="right") / x_sorted.size

    return fitted


def variance_ratio(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute the variance ratio between `a` and `b`, defined as
    .. math:: \frac{\text{Var}(a)}{\text{Var}(b)}

    Parameters
    ----------
    a, b : ArrayLike
        To compute the variance ratio between

    Returns
    -------
    float
        variance ratio
    """
    return np.var(a) / np.var(b)
