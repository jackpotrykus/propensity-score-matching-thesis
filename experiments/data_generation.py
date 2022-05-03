from dataclasses import dataclass
from typing import Tuple, Type, TypeVar

import numpy as np
from numpy import typing as npt


def generate_constant_correlation_matrix(p: int, rho: float) -> np.ndarray:
    """Generate a correlation/covariance matrix with unit entries along the main diagonal, and `rho` elsewhere.

    Note that matrix may not be SPD if `rho < 0`"""
    K = np.zeros((p, p))
    for idx in range(p):
        for jdx in range(p):
            if idx == jdx:
                K[idx, jdx] = 1
            else:
                K[idx, jdx] = rho
    return K

def generate_imbalanced_X(
    size0: int, size1: int, mean0: npt.ArrayLike, mean1: npt.ArrayLike, K: np.ndarray
) -> np.ndarray:
    """Generate `size0` observations with mean `mean0` and covariance `K`, and `size1` observations with mean `mean1`
    and covariance `K`.

    Observations from group 0 constitute the first `size0` rows, and observations from group 1 the last `size1` rows
    """
    p = K.shape[0]

    mu1 = np.zeros(p) + np.asarray(mean0)
    mu2 = np.zeros(p) + np.asarray(mean1)

    X0 = np.random.multivariate_normal(mean=mu1, cov=K, size=size0)
    X1 = np.random.multivariate_normal(mean=mu2, cov=K, size=size1)
    return np.vstack([X0, X1])


_ExperimentDataModel = TypeVar("_ExperimentDataModel", bound="ExperimentDataModel")

@dataclass
class ExperimentDataModel:
    """Handles data generation and organization for the experiments.
    
    THIS IS HIGHLY SPECIFIC TO THE TYPES OF EXPERIMENTS I'M RUNNING.
    IT IS ONLY HERE FOR CONVENIENCE, AND NOT GENERALIZABLE

    Attributes
    ----------
    size0: int
        Number of group 0 (control) observations
    size1: int
        Number of group 1 (treatment) observations
    p: int
        Number of features; the feature matrix `X` has shape `(size0 + size1, p)`
    theta0: float
        Location parameter to a normal distribution, from which group 0 feature means are drawn
    theta1: float
        Location parameter to a normal distribution, from which group 1 feature means are drawn
    rho: float
        Non-diagonal entries of the covariance matrix `K` are filled with `rho`.
        Note that if `rho` is not positive, the covariance matrix is not guaranteed to be SPD.
    mean0: np.ndarray
        Means of the `p` features, for group 0 observations
    mean1: np.ndarray
        Means of the `p` features, for group 1 observations
    K: np.ndarray
        of shape `(p, p)`, the covariance matrix for group 0 *and* group 1
    X: np.ndarray
        of features
    y: np.ndarray
        of outcomes
    z: np.ndarray
        of treatment assignments
    beta: np.ndarray
        The true coefficients used to generate `y` from `X` via linear regression
    true_fit: np.ndarray
        The true value of `X @ beta`
    epsilon: np.ndarray
        Of residuals, sample from the standard Normal distribution
    """
    size0: int
    size1: int

    # Hyperparameters on `X`
    p: int
    theta0: float
    theta1: float
    rho: float

    # Parameters of `X`
    mean0: np.ndarray
    mean1: np.ndarray
    K: np.ndarray

    # The data
    X: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # Outcomes regression
    beta: np.ndarray
    true_fit: np.ndarray
    epsilon: np.ndarray

    @classmethod
    def generate(cls: Type[_ExperimentDataModel], size0: int, size1: int, p: int, theta0: float, theta1: float, rho: float) -> _ExperimentDataModel:
        """Generate new data from the model

        Parameters
        ----------
        size0: int
            Number of group 0 (control) observations to generate
        size1: int
            Number of group 1 (treatment) observations to generate
        p: int
            Number of features to generate; the feature matrix `X` has shape `(size0 + size1, p)`
        theta0: float
            Location parameter to a normal distribution, from which group 0 feature means are drawn
        theta1: float
            Location parameter to a normal distribution, from which group 1 feature means are drawn
        rho: float
            Non-diagonal entries of the covariance matrix `K` are filled with `rho`.
            Note that if `rho` is not positive, the covariance matrix is not guaranteed to be SPD.

        Returns
        -------
        _DataModel
            With data drawn according to model parameters
        """
        # Generate treatment assignments
        z = np.hstack([np.zeros(size0), np.ones(size1)])

        # Generate the mean vectors for `X` using `theta0` and `theta1` as hyperparameters
        mean0 = np.random.normal(loc=theta0, size=p, scale=0.5)
        mean1 = np.random.normal(loc=theta1, size=p, scale=0.5)

        # Generate correlation matrix for `X`
        K = generate_constant_correlation_matrix(p, rho)

        # Now, we can generate `X`
        X = generate_imbalanced_X(size0=size0, size1=size1, mean0=mean0, mean1=mean1, K=K)

        # Sample betas and tau (treatment effet) from normal distrubtion
        beta = np.random.normal(size=p)
        tau = np.random.normal()

        # Calculate true fit, sample residuals, calculate ys
        true_fit = X @ beta + z * tau
        epsilon = np.random.normal(size=size0 + size1)
        y = true_fit + epsilon

        return cls(
            size0=size0,
            size1=size1,
            p=p,
            theta0=theta0,
            theta1=theta1,
            rho=rho,
            mean0=mean0,
            mean1=mean1,
            K=K,
            X=X,
            y=y,
            z=z,
            beta=beta,
            true_fit=true_fit,
            epsilon=epsilon,
        )
