from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, NamedTuple, Union

import numpy as np
from numpy import typing as npt
from scipy import spatial

from matching.dataset import MatchingDataset

"""In sparse representations of graphs, 0 values are taken to mean "no edge" by convention.
In order to prevent the sparse representation from dropping edge weights of 0 (perfect matches),
we clip the edge weights to this small floating point value
"""
DEFAULT_DISTANCE_TOL = 1e-16


class _WeightedEdgeInfo(NamedTuple):
    """`NamedTuple` representing weighted graph edges.
    `Iterable[WeightedEdgeInfo]` can be passed to `networkx.Graph.add_weighted_edges_from`

    Attributes
    ----------
    tid : str
        treatment ID (first node)
    cid : str
        control ID (second node)
    weight : float
        weight of the edge
    """

    tid: str
    cid: str
    weight: float


@dataclass
class _GraphUpdateInfo:
    """Intermediate data structure. Attributes can be used to construct a weighted bipartite graph

    Attributes
    ----------
    tids : ArrayLike
        of treatment ID nodes to be included in the graph, for which `"bipartite" == 1`
    cids : ArrayLike
        of control ID nodes to be included in the graph, for which `"bipartite" == 0`
    edges : Iterable[:class:`WeightedEdgeInfo`]
        Iterable of <tid, cid, edge weight> tuples
    """

    tids: npt.ArrayLike
    cids: npt.ArrayLike
    edges: Iterable[_WeightedEdgeInfo]


# NOTE: type: ignore here because mypy doesn't like abstract dataclasses
# NOTE: see https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class _Distance(metaclass=ABCMeta):
    """Distance metric base class

    Attributes
    ----------
    tol : float, default=`DEFAULT_DISTANCE_TOL`
        A small number to replace 0-weight edges (ie edges of perfect matches) with.
        This is so that the sparse matrix representation does not drop matches.
        See `DEFAULT_DISTANCE_TOL` for details
    """

    tol: float = field(default=DEFAULT_DISTANCE_TOL)

    @abstractmethod
    def _compute(self, data: MatchingDataset) -> _GraphUpdateInfo:
        """Compute edge weights for a `MatchingDataset`

        Parameters
        ----------
        data: :class:`MatchingDataset`
            The dataset to match on

        Returns
        -------
        :class:`_GraphUpdateInfo`
            Intermediate data structure representing a weighted bipartite graph
        """


@dataclass
class Exact(_Distance):
    """:class:`Exact` distance metric. Allows exact matches on all features only to be added as edges to the graph.
    Results are identical to using a `Norm` match with `max_distance=0`, but contains speed optimizations
    to avoid useless computation

    Attributes
    ----------
    tol : float, default=`DEFAULT_DISTANCE_TOL`
        A small number to replace 0-weight edges (ie edges of perfect matches) with.
        This is so that the sparse matrix representation does not drop matches.
        See DEFAULT_DISTANCE_TOL for details
    """

    def _compute(self, data: MatchingDataset) -> _GraphUpdateInfo:
        tids: List[str] = []
        cids: List[str] = []
        edges: List[_WeightedEdgeInfo] = []
        for subdata in data.groupby(list(data.X.columns)):
            subtids = subdata.treatment.ids.values
            subcids = subdata.control.ids.values

            tids.extend(subtids)
            cids.extend(subcids)
            edges.extend([_WeightedEdgeInfo(tid, cid, 1) for tid in subtids for cid in subcids])

        return _GraphUpdateInfo(tids, cids, edges)


@dataclass
class _Tuneable:
    """A baseclass for distance metrics that can use `min_distance` and `max_distance` parameters to tune edge creation.
    Exists for MRO reasons with dataclasses, so that constuctor parameters take the desired order

    Attributes
    ----------
    min_distance : float, default=0
        Minimum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    max_distance : float, default=inf
        Maximum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    """

    min_distance: float = field(default=0)
    max_distance: float = field(default=np.inf)


@dataclass
class _Norm:
    """A baseclass for distance metrics norm-based distance metrics. `p` denotes which Lp Norm to use.
    Exists for MRO reasons with dataclasses, so that constuctor parameters take the desired order

    Attributes
    ----------
    p : Union[int, float]
        Which LpNorm to use. Either an integer value, or infinity
    """

    p: Union[int, float]


@dataclass
class Norm(_Distance, _Tuneable, _Norm):
    """Distance metric for Lp Norms.

    Attributes
    ----------
    p : Union[int, float]
        Which LpNorm to use. Either an integer value, or infinity
    min_distance : float, default=0
        Minimum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    max_distance : float, default=inf
        Maximum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    tol : float, default=`DEFAULT_DISTANCE_TOL`
        A small float. Edge weights of 0 (perfect matches) are replaced with this value, so that the sparse matrix
        representation does not drop perfect matches (normally, 0 = no edge). Does not affect `max_distance`.
        See `DEFAULT_DISTANCE_TOL` for details
    """

    def __post_init__(self) -> None:
        # `p` should either be an `int` or `float("inf")`; `-3.4` for instance is not valid
        if not isinstance(self.p, int):
            assert self.p == float("inf"), 'p must either be an `int` or `float("inf")`'

    def _compute(self, data: MatchingDataset) -> _GraphUpdateInfo:
        # TODO: Explain what is going on here...
        t_data = data.treatment
        c_data = data.control

        # Need non-zero # of treatment and control to make non-trivial bipartite graph
        T, tids = t_data.arrs.X, t_data.arrs.ids
        C, cids = c_data.arrs.X, c_data.arrs.ids

        (n, k), (m, k2) = T.shape, C.shape
        assert k == k2, f"number of columns in T = {k} != {k2} = number of columns in C"

        dists = spatial.distance.cdist(T, C, metric="minkowski", p=self.p)

        # Check that we have a row for every A group and a column for every B group
        assert dists.shape == (n, m), f"expected shape {(n, m)}, got {dists.shape=}"

        idxs = np.ones(shape=(n,)).astype(bool)
        jdxs = np.ones(shape=(m,)).astype(bool)

        idxs, jdxs = np.where(np.logical_and(dists >= self.min_distance, dists <= self.max_distance))
        tids, cids, dists = tids[idxs], cids[jdxs], dists[idxs, jdxs]

        dists = np.clip(dists, a_min=self.tol, a_max=None)
        edges = [_WeightedEdgeInfo(tid, cid, dist) for tid, cid, dist in zip(tids, cids, dists)]
        return _GraphUpdateInfo(tids, cids, edges)


@dataclass
class L1Norm(Norm):
    """L1 :class:`Norm` distance, aka Manhattan distance. Equal to the sum of the absolute differences

    Attributes
    ----------
    p : Union[int, float], default=1
        Which Lp Norm to use. Either an integer value, or infinity
    min_distance : float, default=0
        Minimum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    max_distance : float, default=inf
        Maximum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    tol : float, default=`DEFAULT_DISTANCE_TOL`
        A small float. Edge weights of 0 (perfect matches) are replaced with this value, so that the sparse matrix
        representation does not drop perfect matches (normally, 0 = no edge). Does not affect `max_distance`.
        See `DEFAULT_DISTANCE_TOL` for details
    """

    p: Union[int, float] = field(init=False, default=1)


@dataclass
class L2Norm(Norm):
    """L2 :class:`Norm` distance, aka Euclidean distance. Equal to the square root of the sum of squared differences

    Attributes
    ----------
    p: Union[int, float], default=2
        Which LpNorm to use
    min_distance : float, default=0
        Minimum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    max_distance : float, default=inf
        Maximum allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    tol : float, default=`DEFAULT_DISTANCE_TOL`
        A small float. Edge weights of 0 (perfect matches) are replaced with this value, so that the sparse matrix
        representation does not drop perfect matches (normally, 0 = no edge). Does not affect `max_distance`.
        See `DEFAULT_DISTANCE_TOL` for details
    """

    p: Union[int, float] = field(init=False, default=2)


# Some common aliases
Manhattan = Absolute = L1Norm
Euclidean = L2Norm


# TODO: Refactor the below
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matching import preprocessing
from matching._typing import sklearn_types as skt

@dataclass
class Mahalanobis(Norm):
    """Mahalanobis distance. This is equivalent to standard-scaling all vectors to have mean 0 and variance 1, and then
    using :class:`L2Norm` distance on the standard-scaled differences

    Attributes
    ----------
    p: Union[int, float]
        Which LpNorm to use. Either an integer value, or infinity. This is pre-set to 2 for Mahalanobis norm.
    max_distance: float
        Max allowable edge weight. Can be used as a tuning parameter for identifying similar groups
    tol: float
        A small float. Edge weights of 0 (perfect matches) are replaced with this value, so that the sparse matrix
        representation does not drop perfect matches (normally, 0 = no edge). Does not affect max_distance
        See DEFAULT_DISTANCE_TOL for details
    """

    model: skt._Preprocessor = field(init=False)
    p: Union[int, float] = field(init=False, default=2)

    def _compute(self, data: MatchingDataset) -> _GraphUpdateInfo:
        self.model = StandardScaler()
        scaled_X = self.model.fit_transform(data.arrs.X)
        return super()._compute(MatchingDataset(scaled_X, data.z, data.ids))


@dataclass
class PropensityScore(L1Norm):
    model: skt._Classifier = field(default_factory=lambda: LogisticRegression(penalty="none"))
    caliper: float = field(default=np.inf)
    use_logit: bool = True
    tol: float = field(default=DEFAULT_DISTANCE_TOL)

    p: int = field(init=False, default=1)
    max_distance: float = field(init=False, default=np.inf)

    def _compute(self, data: MatchingDataset) -> _GraphUpdateInfo:
        scores = preprocessing.propensity_score(data.arrs.X, data.arrs.z, model=self.model, use_logit=self.use_logit)

        self.max_distance = self.caliper * np.std(scores)
        return super()._compute(MatchingDataset(scores, data.z, data.ids))
