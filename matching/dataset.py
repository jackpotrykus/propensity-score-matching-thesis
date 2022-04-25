from dataclasses import dataclass
from typing import Hashable, Iterator, NamedTuple, Optional, Iterable, TypeVar, Union

from numpy import typing as npt
import numpy as np
import pandas as pd


class _MatchingDatasetNDArrays(NamedTuple):
    """Container for the NDArray equivalent of each element of a :class:`MatchingDataset`.
    Note that non-numeric columns of `X` are one-hot encoded.

    Attributes
    ----------
    X: ndarray
        of features, with non-numeric columns one-hot encoded with `drop_first=True`
    z: ndarray
        of boolean treatment assignments
    ids: ndarray
        of string IDs
    """

    X: np.ndarray
    z: np.ndarray
    ids: np.ndarray


"""TypeVar for :class:`MatchingDataset`, for methods which return another :class:`MatchingDataset`"""
_MatchingDataset = TypeVar("_MatchingDataset", bound="MatchingDataset")


@dataclass(init=False, eq=False)
class MatchingDataset:
    """Container that validates inputs to be used in bipartite matching

    Attributes
    ----------
    X: DataFrame
        of features to match on
    z: Series
        of boolean treatment assignments
    ids: Series
        of observation IDs. Note that these are coerced to `str`
    """

    X: pd.DataFrame
    z: pd.Series
    ids: pd.Series

    def __init__(
        self,
        X: npt.ArrayLike,
        z: npt.ArrayLike,
        ids: Optional[npt.ArrayLike] = None,
        treatment_encoded_as: Hashable = True,
    ):
        """
        Parameters
        ----------
        X: ArrayLike
            of features to match on. Will be converted to a `pd.DataFrame`
        z: ArrayLike
            of boolean treatment assignments. Will be converted to a `pd.Series`
        ids: ArrayLike
            of observation IDs. Will be converted to a `pd.Series`, and coerced to type `str`
        """
        self.X = pd.DataFrame(X).copy().reset_index(drop=True)
        self.z = pd.Series(np.asarray(z) == treatment_encoded_as, name="is_treatment").copy()

        # Auto-generate IDs, if not supplied
        if ids is None:
            self.ids = pd.Series(np.arange(self.nobs)).astype(str)
        else:
            self.ids = pd.Series(np.asarray(ids), name="id").copy().astype(str)

        # Number of rows must be consistent across all attributes
        if not self.X.shape[0] == self.z.shape[0] == self.ids.shape[0]:
            raise ValueError(f"Dimension mismatch. {self.X.shape=}, {self.z.shape=}, {self.ids.shape=}")

    def __eq__(self, __o: object) -> bool:
        """Equals logic: are two datasets the same?

        Parameters
        ----------
        __o: object
            for comparison

        Returns
        -------
        bool
            Indicating if all elements are equal
        """
        if isinstance(__o, MatchingDataset):
            return self.X.equals(__o.X) and self.z.equals(__o.z) and self.ids.equals(__o.ids)
        return False

    @property
    def nobs(self) -> int:
        """Total number of observations associated with this dataset"""
        return self.X.shape[0]

    @property
    def treatment(self: _MatchingDataset) -> _MatchingDataset:
        """Dataset of just the treatment group"""
        return self._subset_by_indexer(self.z.values.astype(bool))

    @property
    def control(self: _MatchingDataset) -> _MatchingDataset:
        """Dataset of just the control group"""
        # NOTE: mypy ignoring the below ... thinks it might become an "ExtensionArray"
        return self._subset_by_indexer(~self.z.values.astype(bool))

    @property
    def frame(self) -> pd.DataFrame:
        """Dataframe representation of this dataset"""
        to_concat = [self.X, self.z, self.ids]
        return pd.concat([obj.reset_index(drop=True) for obj in to_concat], axis=1).reset_index(drop=True)

    @property
    def arrs(self) -> _MatchingDatasetNDArrays:
        """Numpy array versions of `X`, `z`, and `ids`. Non-numeric columns of `X` are one-hot encoded"""
        Xarr = pd.get_dummies(self.X, drop_first=True).values
        zarr = self.z.values.astype(bool)
        idsarr = self.ids.values.astype(str)
        return _MatchingDatasetNDArrays(Xarr, zarr, idsarr)

    def _subset_by_indexer(self: _MatchingDataset, indexer: npt.ArrayLike) -> _MatchingDataset:
        """Subset this :class:`_MatchingDataset` by supplying a boolean indexing array

        Parameters
        ----------
        indexer: ArrayLike
            of boolean values, indicating which rows to keep

        Returns
        -------
        :class:`_MatchingDataset`
            Subsetted
        """
        sub_X = self.X.loc[indexer, :]
        sub_z = self.z.loc[indexer]
        sub_ids = self.ids.loc[indexer]
        return type(self)(sub_X, sub_z, sub_ids)

    def groupby(
        self: _MatchingDataset, by: Union[Hashable, Iterable[Hashable], npt.ArrayLike], *args, **kwargs
    ) -> Iterator[_MatchingDataset]:
        """Works like `pd.DataFrame.groupby`, except it yields :class:`_MatchingDataset` instead

        Parameters
        ----------
        by: Union[Hashable, Iterable[Hashable], npt.ArrayLike]
            A label, sequence of labels, or indexing array to supply to `pd.DataFrame.groupby`
        *args, **kwargs
            Passed to `pd.DataFrame.groupby`

        Yields
        -------
        :class:`_MatchingDataset`
            Note that no copy is made
        """
        for _, subdf in self.frame.groupby(by, *args, **kwargs):
            subX = subdf.loc[:, self.X.columns]
            subz = subdf.loc[:, self.z.name]
            subids = subdf.loc[:, self.ids.name]

            data = type(self)(subX, subz, subids)
            yield data

    def copy(self: _MatchingDataset, deep: bool = False) -> _MatchingDataset:
        """Make a copy of this dataset.

        Parameters
        ----------
        deep: bool
            `False` by default. Whether or not to make a deep copy

        Returns
        -------
        :class:`_MatchingDataset`
            copy
        """
        X = self.X.copy(deep)
        z = self.z.copy(deep)
        ids = self.ids.copy(deep)
        return type(self)(X, z, ids)

    def summary(self) -> pd.DataFrame:
        """Print a summary of the matching :class:`_MatchingDataset`"""
        return pd.concat(
            [
                self.frame.groupby(self.z.name).describe(include=np.number).T,
                self.frame.groupby(self.z.name).describe(include=np.object_).T,
            ],
            axis=0,
        )
