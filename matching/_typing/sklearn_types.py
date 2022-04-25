from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class _Fittable(Protocol):
    """Anything with a `fit()` method. Most `sklearn` estimators satisfy this"""

    def fit(self, X, y) -> Any:
        ...


@runtime_checkable
class _Regressor(_Fittable, Protocol):
    """Any `_Fittable` with a `predict` method"""

    def predict(self, X) -> Any:
        ...


@runtime_checkable
class _Classifier(_Regressor, Protocol):
    """Any `_Fittable` with a `predict_proba` method"""

    def predict_proba(self, X) -> Any:
        ...


@runtime_checkable
class _Preprocessor(_Fittable, Protocol):
    """Any `_Fittable` with `transform` and `fit_transform` methods"""

    def transform(self, X) -> Any:
        ...

    def fit_transform(self, X, y=None, **fit_params) -> Any:
        ...

    def inverse_transform(self, X) -> Any:
        ...
