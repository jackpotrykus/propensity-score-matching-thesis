from collections import defaultdict
from dataclasses import dataclass, field
import enum
from typing import Dict, Hashable, Iterable, Iterable, List, Optional, TypeVar, Union

import networkx as nx
from numpy import typing as npt
import numpy as np
from scipy import sparse

from matching import graph_utils as gu
from matching.dataset import MatchingDataset
from matching.distance import _Distance


class NotBipartiteError(Exception):
    """Raised if the graph is not bipartite"""


@dataclass
class BipartiteGraphParameters:
    """Container of various parameters relevant to initializing, plotting, and otherwise working with bipartite graphs
    using networkx

    Attributes
    ----------
    init_weight: float
        The default weight between nodes in the naive graph, before ``set_edges`` is called
    treatment_bipartite_attr: Hashable
        What the "bipartite" attribute for will be set to for each treatment node in the graph
    control_bipartite_attr: Hashable
        What the "bipartite" attribute for will be set to for each control node in the graph
    treatment_color: str
        Color to plot treatment nodes as
    control_color: str
        Color to plot control nodes as
    match_group_col: str
        Name of column to add to the graph data, indicating which match the observation belongs to
    """

    init_weight: float = field(default=1)
    treatment_bipartite_attr: Hashable = field(default=1)
    control_bipartite_attr: Hashable = field(default=0)
    treatment_color: str = field(default="red")
    control_color: str = field(default="blue")
    match_group_col: str = field(default="match_group")


class MatchingMethod(enum.Enum):
    """Enum for valid matching methods, with some aliases"""

    FAST = GREEDY = enum.auto()
    HUNGARIAN = OPTIMAL = KUHN = MUNKRES = enum.auto()


"""``TypeVar`` for :class:`MatchingGraph`, since many methods return ``self`` to enable method chaining"""
_MatchingGraph = TypeVar("_MatchingGraph", bound="MatchingGraph")


@dataclass(init=False)
class MatchingGraph:
    """:class:`MatchingGraph` offers a simple framework to
    - calculate a distance metric between two sides of a bipartite graph,
    - iteratively filter this graph by enforcing other constraints on nodes and edges,
    - calculate a "matched" subset of this graph via greedy, grouping, or optimal matching strategies, and
    - compare the balance between input and matched datasets.

    At construction, the user supplies three ``ArrayLike``s:
    - ``X``, an array of features to match on;
    - ``z``, an array of binary treatment assignments; and
    - ``ids``, an optional array of observation IDs. If not supplied, default IDs are enumerated from 0 upwards.

    These inputs are parsed into the attribute ``input_data``. The user can then proceed to ``set_edges`` by supplying
    a :class:`_Distance` measure, optionally specifying specific columns of ``X`` to ``include`` or ``exclude`` from the
    distance calculation.

    Once the edges are calculated, the user can then use one of the matching strategies provided to "match" their
    dataset. Alternatively, the user can proceed to iteratively filter the graph further, via the ``filter_edges`` and
    ``filter_nodes``.
    - ``filter_edges`` allows the user to enforce additional distance metrics on the data, while preserving the edge
    weights from ``set_edges``. For example, a user may calculate edge weights on a propensity score, but then require
    an exact match on certain categorical features, as well as a cutoff for the Mahalanobis distance between the feature
    vectors.
    - ``filter_nodes`` allows the user to filter nodes via degree, or label. For example, the user may wish to only keep
    nodes of degree at least 3, after an initial pruning on ``Mahalanobis`` distance.


    Attributes
    ----------
    input_data: :class:`MatchingDataset`
        The data which is supplied to the constructor (X, z, ids) is parsed into an ``input_data`` object, with attributes
        ``X``, ``z``, and ``ids`. You can convert this to a ``pd.DataFrame`` via ``<MatchingGraph>.input_data.frame``
    graph_data: :class:`MatchingDataset`
        Just the data for the nodes currently present on the graph. Has all the same properties as ``input_data``. Notes
        that this is a derived property that is re-calculated each time it is accessed.
    match_data: :class:`MatchingDataset`
        This is the same as ``graph_data``, except it adds an additional column to ``X``: "match_group", an integer number
        indicating which matched subgraph the match belongs to
    graph: nx.Graph
        Bipartite graph, where nodes represent observation IDs and edge weights correspond to the cost of matching
    distance: :class:`_Distance`
        The distance measure used to calculate the edge weights on the graph
    _params: :class:`BipartiteGraphParameters`
        Private attribute that provides a single container for default arguments to pass to graph functions
    """

    # These parameters are set in the call to ``_init__``
    input_data: MatchingDataset = field(init=False)
    _params: BipartiteGraphParameters = field(init=False)

    # These parameters are set on the call to ``set_edges``
    graph: nx.Graph = field(init=False)
    distance: _Distance = field(init=False)

    def __init__(
        self, X: npt.ArrayLike, z: npt.ArrayLike, ids: Optional[npt.ArrayLike] = None, *, init_naive_graph: bool = False
    ) -> None:
        """Initialize a :class:`MatchingGraph` object with necessary data

        Parameters
        ----------
        X: ArrayLike
            of features to match on. Will be converted to a ``pd.DataFrame``
        z: ArrayLike
            of boolean treatment assignments. Will be converted to a ``pd.Series``
        ids: ArrayLike
            of observation IDs. Will be converted to a ``pd.Series``, and coerced to type ``str``
        init_naive_graph: bool
            Keywoard argument only. ``False`` by default; whether or not to initialize a "naive", fully-connected graph on init.
            This can be time-consuming. NOTE: if this is ``False``, the user MUST call ``set_edges`` before doing any filtering!
            Otherwise the graph will remain empty
        """
        self.input_data = MatchingDataset(X, z, ids)

        # NOTE: These would be default_factory if using a builder...
        self._params = BipartiteGraphParameters()

        self.graph = nx.Graph()
        if init_naive_graph:
            # Initialize graph with nodes, edges all == 1
            tids = self.input_data.treatment.arrs.ids
            cids = self.input_data.control.arrs.ids
            self.graph.add_nodes_from(tids, bipartite=self._params.treatment_bipartite_attr)
            self.graph.add_nodes_from(cids, bipartite=self._params.control_bipartite_attr)
            self.graph.add_weighted_edges_from((tid, cid, self._params.init_weight) for tid in tids for cid in cids)

    @property
    def graph_data(self) -> MatchingDataset:
        """The :class:`MatchingDataset` for just the data currently present on the graph"""
        graph_indexer = self.input_data.ids.isin(set(self.graph.nodes))
        return self.input_data._subset_by_indexer(graph_indexer)

    @property
    def match_data(self) -> MatchingDataset:
        """The :class:`MatchingDataset` for just the data currently present on the graph.
        An additional column to "X" is added: "match_group", indicating which subgraph of ``self.subgraphs`` the match belongs to
        This is the data compared against the input data when computing balance metrics.
        """
        data = self.input_data.copy()
        data.X[self._params.match_group_col] = -1

        for idx, H in enumerate(self.subgraphs):
            subgraph_ids = set(H.nodes)

            # If the subgraph has only one node, it's not a match
            if len(subgraph_ids) > 1:
                subgraph_indexer = data.ids.isin(set(H.nodes))
                data.X.loc[subgraph_indexer, self._params.match_group_col] = idx

        graph_indexer = data.X[self._params.match_group_col] >= 0
        return data._subset_by_indexer(graph_indexer)

    @property
    def subgraphs(self) -> List[nx.Graph]:
        """Get the connected component subgraphs of ``self.graph`` in a list.
        NOTE: No copy is made here, so you will need to make a copy of each subgraph if you don't want changes to be
        reflected in the graph
        """
        return gu.get_connected_subgraphs(self.graph)

    @property
    def node_colors(self) -> List[str]:
        """Passed as ``node_color`` to ``networkx`` drawing functions"""
        is_treatment = lambda data: data == self._params.treatment_bipartite_attr
        get_node_color = lambda data: self._params.treatment_color if is_treatment(data) else self._params.control_color
        return [get_node_color(data) for _, data in self.graph.nodes(data="bipartite")]  # type: ignore

    def _parse_include_exclude(
        self,
        include: Optional[Iterable[Hashable]],
        exclude: Optional[Iterable[Hashable]],
        *,
        use_graph_data: bool,
    ) -> MatchingDataset:
        """Some methods, such as ``set_edges`` and ``filter_edges``, allow the user to specify specifically which columns
        of ``X`` to use in the distance calculation. This helper method parses these inputs to get the requested data only

        Parameters
        ----------
        include: Optional[Iterable[Hashable]]
            Features to __include__ in the distance calculation, ``None`` by default.
            When ``None``, all features from ``self.input_data.X`` are included.
            When ``Hashable`` or ``Iterable[Hashable]``, distance only uses corresponding columns of ``self.input_data.X``
        exclude: Optional[Iterable[Hashable]]
            Features to __exclude__ in the distance calculation, ``None`` by default.
            When ``None``, no features from ``self.input_data.X`` are excluded.
            When ``Hashable`` or ``Iterable[Hashable]``, distance drops corresponding columns of ``self.input_data.X``
        use_graph_data: bool
            Whether or not to use the graph data as opposed to the input data

        Returns
        -------
        :class:`MatchingDataset`
            With columns of ``X`` as specified by ``include`` and ``exclude``
        """

        # Easy win if both arguments are ``None``: just return ``self.input_data.X``
        data = self.graph_data if use_graph_data else self.input_data

        if include is None and exclude is None:
            return data
        elif include is None:
            assert exclude is not None
            subX = data.X.loc[:, [c for c in self.input_data.X.columns if c not in exclude]]
        elif exclude is None:
            subX = data.X.loc[:, include]
        else:
            include_set = set(include)
            exclude_set = set(exclude)

            keep_columns = list(include_set.difference(exclude_set))
            subX = data.X.loc[:, keep_columns]

        return MatchingDataset(subX, data.z, data.ids)

    def set_edges(
        self: _MatchingGraph,
        distance: _Distance,
        *,
        include: Optional[Iterable[Hashable]] = None,
        exclude: Optional[Iterable[Hashable]] = None,
        allow_new_node: bool = True,
        allow_new_edge: bool = True,
    ) -> _MatchingGraph:
        """Set the edges of the bipartite graph to have weights determined by ``distance``

        Parameters
        ----------
        distance: :class:`_Distance`
            Distance measure to use
        include: Optional[Iterable[Hashable]]
            Features to __include__ in the distance calculation, ``None`` by default.
            When ``None``, all features from ``self.input_data.X`` are included.
            When ``Hashable`` or ``Iterable[Hashable]``, distance only uses corresponding columns of ``self.input_data.X``
        exclude: Optional[Iterable[Hashable]]
            Features to __exclude__ in the distance calculation, ``None`` by default.
            When ``None``, no features from ``self.input_data.X`` are excluded.
            When ``Hashable`` or ``Iterable[Hashable]``, distance drops corresponding columns of ``self.input_data.X``
        allow_new_node: bool
            Whether to allow new nodes to be added to the graph, ``True`` by default.
        allow_new_edge: bool
            Whether to allow new edges to be added to the graph, ``True`` by default.

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The fitted graph is available in ``self.graph``
        """
        data = self._parse_include_exclude(include, exclude, use_graph_data=not allow_new_node)

        # Update the ``distance`` property in self
        self.distance = distance

        # Compute pairwise differences
        graph_info = self.distance._compute(data)

        # Checks that each node already exists, or ``allow_new_node`` was specified
        node_check = lambda ei: allow_new_node or (ei.tid in self.graph and ei.cid in self.graph)
        # Checks that the edge already exists, or ``allow_new_edge`` was specified
        edge_check = lambda ei: allow_new_edge or self.graph.has_edge(ei.tid, ei.cid)

        # Update weights and return ``self``
        if allow_new_node:
            # Set the bipartite attr for each node
            self.graph.add_nodes_from(graph_info.tids, bipartite=self._params.treatment_bipartite_attr)
            self.graph.add_nodes_from(graph_info.cids, bipartite=self._params.control_bipartite_attr)

        self.graph.add_weighted_edges_from(ei for ei in graph_info.edges if node_check(ei) and edge_check(ei))
        return self

    def filter_edges(
        self: _MatchingGraph,
        distance: _Distance,
        *,
        include: Optional[Iterable[Hashable]] = None,
        exclude: Optional[Iterable[Hashable]] = None,
        drop_isolated: bool = True,
    ) -> _MatchingGraph:
        """Filter edges of the bipartite graph determined by ``distance``.
        This only has any effect if supplied ``distance`` measure has non-zero ``min_distance`` or finite ``max_distance``.

        By default, this __keeps__ edges which also exist in the filter graph.

        Parameters
        ----------
        distance: _Distance
            Distance measure to use
        include: Optional[Iterable[Hashable]]
            Features to __include__ in the distance calculation, ``None`` by default.
            When ``None``, all features from ``self.input_data.X`` are included.
            When ``Hashable`` or ``Iterable[Hashable]``, distance only uses corresponding columns of ``self.input_data.X``
        exclude: Optional[Iterable[Hashable]]
            Features to __exclude__ in the distance calculation, ``None`` by default.
            When ``None``, no features from ``self.input_data.X`` are excluded.
            When ``Hashable`` or ``Iterable[Hashable]``, distance drops corresponding columns of ``self.input_data.X``
        drop_isolated: bool
            ``True`` by default. Whether or not to remove any nodes with 0 connected edges, following filtering

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The filtered graph is available in ``self.graph``
        """
        data = self._parse_include_exclude(include, exclude, use_graph_data=True)

        # Compute which edges to keep via the distance metric
        graph_info = distance._compute(data)

        # Create a set of these edges for O(1) lookup
        filtered_edges = set((u, v) for u, v, *_ in graph_info.edges)

        # Order-agnostic check if the edge tuple is in the ``filtered_edges`` set
        is_in_filtered = lambda edge: (edge[0], edge[1]) in filtered_edges or (edge[1], edge[0]) in filtered_edges

        # Now, we have a to examine the contents of ``self.graph``
        # If the graph is empty, we should add naively-weighted edges to the graph,
        self.graph.remove_edges_from(edge for edge in self.graph.edges if not is_in_filtered(edge))

        # If ``drop_isolated``, we filter nodes that have degree 1. Can happen when all edges get removed
        return self.filter_nodes(min_degree=1) if drop_isolated else self

    def filter_nodes(
        self: _MatchingGraph,
        *,
        min_degree: int = 0,
        max_degree: Union[int, float] = np.inf,
        keep_nodes: Optional[Iterable[str]] = None,
        drop_nodes: Optional[Iterable[str]] = None,
    ) -> _MatchingGraph:
        """Filter nodes of the graph

        Parameters
        ----------
        min_degree: int
            0 by default. Only keep nodes of degree (# of connected edges) at least this quantity
        max_degree: Union[int, float]
            ``np.inf`` by default. Only keep nodes of degree (# of connected edges) at most this quantity
        keep_nodes: Optional[Iterable[str]]
            If supplied, a set of node labels to keep; all other nodes are dropped. ``None`` by default
        drop_nodes: Optional[Iterable[str]]
            If supplied, a set of node labels to drop; all other nodes are kept. ``None`` by default

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The filtered graph is available in ``self.graph``
        """
        # Only filter by degree if non-trivial arguments supplied
        if min_degree > 0 or max_degree < np.inf:
            node_to_degree = self.graph.degree([n for n in self.graph.nodes])
            self.graph.remove_nodes_from(
                node for node, degree in node_to_degree if not min_degree <= degree <= max_degree
            )

        # Filter nodes by label, if ``keep_set`` supplied
        if keep_nodes is not None:
            keep_set = set(keep_nodes)
            self.graph.remove_edges_from(node for node in self.graph.nodes if node not in keep_set)

        if drop_nodes is not None:
            drop_set = set(drop_nodes)
            self.graph.remove_edges_from(node for node in self.graph.nodes if node in drop_set)

        return self

    def filter_subgraphs(
        self,
        *,
        min_order: int = 2,
        max_order: Union[int, float] = np.inf,
        min_size: int = 1,
        max_size: Union[int, float] = np.inf,
        min_treatment: int = 1,
        max_treatment: Union[int, float] = np.inf,
        min_control: int = 1,
        max_control: Union[int, float] = np.inf,
        max_control_to_treatment_ratio: float = np.inf,
        max_treatment_to_control_ratio: float = np.inf,
    ):
        """Filter the subgraphs of ``self.graph`` via a variety of metrics.

        This approach is most common when using Coarsened Exact Matching (CEM) or other Exact Matching constraints.
        In this case, the maximum allowable edge weight is 0 -- a perfect match -- and so "matching" in a 1:k fashion
        is arbitrary: you are just giving up data, when match qualities are all identical.

        This method is also useful in non-exact matching scenarios, where the max_distance has been tuned to be
        sufficiently small such that the subgraphs are indeed identifying highly similar observations on their own,
        and there is no desire for a 1:k match

        Parameters
        ----------
        min_order: int
            Minimum number of nodes in the subgraph for the matches to be included. Default is 2
        max_order: int
            Maximum number of nodes in the subgraph for the matches to be included. Default is ``np.inf``
        min_size: int
            Minimum number of edges in the subgraph for the matches to be included. Default is 1
        max_size: int
            Maximum number of edges in the subgraph for the matches to be included. Default is ``np.inf``
        min_treatment: int
            Minimum number of treatment nodes in the subgraph for the matches to be included. Default is 1
        max_treatment: Union[int, float]
            Maximum number of treatment nodes in the subgraph for the matches to be included. Default is ``np.inf``
        min_control: int
            Minimum number of control nodes in the subgraph for the matches to be included. Default is 1
        max_control: Union[int, float]
            Maximum number of control nodes in the subgraph for the matches to be included. Default is ``np.inf``
        max_control_to_treatment_ratio: float
            Maximum ratio of control: treatment within a subgraph for the matches to be included. Default is ``np.inf``
        max_treatment_to_control_ratio: float
            Maximum ratio of treatment: control within a subgraph for the matches to be included. Default is ``np.inf``

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The fitted filtered is available in ``self.graph``
        """
        assert min_order >= 2, "min_order must be greater than 2"
        assert min_size >= 1, "min_size must be greater than 1"
        assert max_order >= min_order, f"{max_order=} is less than {min_order=}"
        assert max_size >= min_size, f"{max_size=} is less than {min_size=}"

        keep_nodes = set()
        subgraphs = gu.get_connected_subgraphs(self.graph)
        for H in subgraphs:
            order = len(H.nodes)
            if not min_order <= order <= max_order:
                continue

            size = len(H.edges)
            if not min_size <= size <= max_size:
                continue

            n_treatment = sum([1 for _, data in H.nodes(data="bipartite") if data == self._params.treatment_bipartite_attr])  # type: ignore
            if not min_treatment <= n_treatment <= max_treatment:
                continue

            n_control = order - n_treatment
            if not min_control <= n_control <= max_control:
                continue

            control_to_treatment_ratio = n_control / n_treatment
            if control_to_treatment_ratio > max_control_to_treatment_ratio:
                continue

            treatment_to_control_ratio = n_treatment / n_control
            if treatment_to_control_ratio > max_treatment_to_control_ratio:
                continue

            keep_nodes = keep_nodes.union(H.nodes)

        drop_nodes = set(self.graph.nodes).difference(keep_nodes)
        self.graph.remove_nodes_from(drop_nodes)

        return self

    def _filter_graph_by_match_dict(self: _MatchingGraph, match_dict: Dict[str, List[str]]) -> _MatchingGraph:
        """Subset the graph to only contain edges present in the ``match_dict``

        Parameters
        ----------
        match_dict: Dict[str, List[str]]
            Dictionary of <tid>, <one or more cids> matches

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The matched graph is available in ``self.graph``
        """
        # Ensure valid matches -- all keys must be treatment, and all values must control
        is_treatment = lambda _id: _id in set(self.graph_data.treatment.ids)
        assert all(is_treatment(tid) and not any(is_treatment(cid) for cid in cids) for tid, cids in match_dict.items())

        # Set of nodes to keep and a lambda which checks if any node is a member
        keep_nodes = set(_id for tid, cids in match_dict.items() for _id in (tid, *cids))
        self.graph.remove_nodes_from([n for n in self.graph.nodes if n not in keep_nodes])

        # Set of edges to keep and a lambda which checks if any edge is a member
        keep_edges = set((tid, cid) for tid, cids in match_dict.items() for cid in cids)
        isin_keep_edges = lambda edge: (edge[0], edge[1]) in keep_edges or (edge[1], edge[0]) in keep_edges
        self.graph.remove_edges_from(e for e in self.graph.edges if not isin_keep_edges(e))

        return self

    def match(
        self,
        *,
        n_match: int = 1,
        min_match: int = 1,
        replace: bool = False,
        method: Union[str, MatchingMethod] = "greedy",
    ):
        """Conduct matching based on the current nodes and edges in ``self.graph``.

        This method is a wrapper around other methods of this class, and dispatches to them according to ``method``.
        All parameter descriptions below come with the caveat that they are simply ignored where not applicable.
        See documentation for other ``match_*`` methods for details on what is used where, and how

        Parameters
        ----------
        n_match: int
            Maximum number of matches per treatment group observation. 1 by default
        min_match: int
            Minimum number of matches per treatment group observation. Patients with less than this amount of matches
            are dropped from the result
        replace: bool = False
            Whether or not to conduct matching with replacement. If ``True``, then two treatment group observations can
            match with the same control group observation. ``False`` by default
        method: Union[str, MatchingMethod]
            Matching method to use; "greedy" by default. Valid options are (case-insensitive):
            - "greedy" == "fast"
            - "optimal" == "hungarian" == "kuhn" == "munkres"

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The matched graph is available in ``self.graph``
        """
        if isinstance(method, str):
            method = MatchingMethod[method.upper()]

        if method is MatchingMethod.GREEDY:
            return self.match_greedy(n_match=n_match, min_match=min_match, replace=replace)
        elif method is MatchingMethod.OPTIMAL:
            return self.match_optimal(n_match=n_match, min_match=min_match, replace=replace)
        else:
            raise NotImplementedError("Requested method has not yet been implemented")

    def match_optimal(
        self: _MatchingGraph, *, n_match: int = 1, min_match: int = 1, replace: bool = False
    ) -> _MatchingGraph:
        """Conduct matching based on the current nodes and edges in ``self.graph`` via the Hungarian algorithm.
        The Hungarian algorithm solves the _assignment problem_, performing a simulatenous bipartite matching which
        minimizes the total sum of the edge weights.


        Parameters
        ----------
        n_match: int
            Maximum number of matches per treatment group observation. 1 by default
        min_match: int
            Minimum number of matches per treatment group observation. Patients with less than this amount of matches
            are dropped from the result
        replace: bool = False
            Whether or not to conduct matching with replacement. If ``True``, then two treatment group observations can
            match with the same control group observation. ``False`` by default

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The matched graph is available in ``self.graph``

        Notes
        -----
        This algorithm runs in cubic time, with respect to the number of edges on the graph. As such, users should
        choose a suitable ``max_distance`` for the distance measure, or make use of ``filter_edges`` and/or ``filter_nodes``
        to limit the size (# of edges) of the graph, before running.

        ``networkx`` and ``scipy`` are a bottleneck to performance -- their implementations of this minimum weight bipartite
        matching algorithm are each written in pure python.
        """
        preliminary_matches = defaultdict(list)

        # Copy the graph since we will be modifying it heavily
        G: nx.Graph = self.graph.copy()  # type: ignore

        # O(1) lookups for treatment group membership
        tids = set(self.graph_data.treatment.ids.values)

        # Generates a "dup" ID for 1-many matching that we can revert
        ID_DUP_CHAR = "_"
        dup_id = lambda _id, n: str(_id) + ID_DUP_CHAR * n
        undup_id = lambda _id: str(_id).strip(ID_DUP_CHAR)

        # Checks if an ID is a treatment ID
        is_treatment = lambda _id: undup_id(_id) in tids

        # Add ``n_match - 1`` duplicate nodes for each treatment node
        for n in range(1, n_match):
            for id1, id2, weight in self.graph.edges.data("weight"):  # type: ignore
                if is_treatment(id1):
                    G.add_edge(dup_id(id1, n), id2, weight=weight)
                elif is_treatment(id2):
                    G.add_edge(dup_id(id2, n), id1, weight=weight)
                else:
                    raise NotBipartiteError

        # Identify distnct connected component subgraphs
        for H in gu.get_connected_subgraphs(G):
            tid_degrees = H.degree([n for n in H.nodes if is_treatment(n)])
            cid_degrees = H.degree([n for n in H.nodes if not is_treatment(n)])

            # If a connected component subgraph does not have a treatment or control ID, we cannot match
            if len(tid_degrees) == 0 or len(cid_degrees) == 0:
                # ... but also, it should be an isolated node, otherwise the graph is not bipartite. So check that here
                if not max(len(tid_degrees), len(cid_degrees)) == 1:
                    raise NotBipartiteError
                continue

            # As a convention, if we cannot perform a full match, we will drop the nodes with the fewest matches
            tid_order = np.array([n for n, _ in sorted(tid_degrees, key=lambda tup: tup[1], reverse=True)])
            cid_order = np.array([n for n, _ in sorted(cid_degrees, key=lambda tup: tup[1], reverse=True)])

            # As a convention, we set the rows to be the smaller of the two bipartite sets
            T_IS_ROW = len(tid_order) <= len(cid_order)
            if T_IS_ROW:
                ro, co = tid_order, cid_order
            else:
                ro, co = cid_order, tid_order

            # Convert the subgraph into a biadjacency matrix
            bm = nx.algorithms.bipartite.biadjacency_matrix(H, row_order=ro, column_order=co)
            if not isinstance(bm, (sparse.csr_array, sparse.csr_matrix)):
                raise nx.NetworkXError(f"Failed to create biadjacency matrix, got {type(bm)=}")

            # NOTE: Here, we handle the case where a full matching is not possible
            # scipy's ``min_weight_full_bipartite_matching`` does not work unless a full matching is possible, it throws
            # a ValueError. So we have to first find the maximum bipartite matching and limit the biadjacency matrix to
            # just the matched data.
            # Returns row-indexing array with -1 for nodes excluded from maximum bipartite matching
            indexer = sparse.csgraph.maximum_bipartite_matching(bm.transpose())
            bm = bm[indexer >= 0, :]

            # Get matched indices from Hungarian algorithm. Tranpose according to ``T_IS_ROW``
            tdxs, cdxs = sparse.csgraph.min_weight_full_bipartite_matching(bm if T_IS_ROW else bm.transpose())

            # Convert matched indices into matched IDs
            matched_tids, matched_cids = tid_order[tdxs], cid_order[cdxs]

            # De-dupify IDs and add match to dictionary
            for tid, cid in zip(matched_tids, matched_cids):
                deduped_id = undup_id(tid)
                preliminary_matches[deduped_id].append(cid)

            # Remove edges from the temporary graph
            G.remove_edges_from((tid, cid) for tid, cid in zip(matched_tids, matched_cids))

            # If ``not replace``, we should remove the cid node altogether
            if not replace:
                G.remove_nodes_from(cid for cid in matched_cids)

        # Drop keys with less than ``min_match`` matches
        optimal_matches = {tid: cids for tid, cids in preliminary_matches.items() if len(cids) >= min_match}
        return self._filter_graph_by_match_dict(optimal_matches)

    def match_greedy(
        self: _MatchingGraph, *, n_match: int = 1, min_match: int = 1, replace: bool = False
    ) -> _MatchingGraph:
        """Conduct matching based on the current nodes and edges in ``self.graph`` via a greedy approach.
        The greedy algorithm works as follows:
        1. Sort graph edges by weight, from smallest (closest match) to largest
        2. For each edge,
            a. if the treatment observation already has ``n_match`` matches, skip; otherwise,
            b. if ``replace = True`` __OR__ the control observation has not yet been matched, add this control as a match
            to the treatment observation
        3. Remove matches with fewer than ``min_match`` matched control observations

        Parameters
        ----------
        n_match: int
            Maximum number of matches per treatment group observation. 1 by default
        min_match: int
            Minimum number of matches per treatment group observation. Patients with less than this amount of matches
            are dropped from the result
        replace: bool = False
            Whether or not to conduct matching with replacement. If ``True``, then two treatment group observations can
            match with the same control group observation. ``False`` by default

        Returns
        -------
        :class:`_MatchingGraph`
            same instance. This allows method chaining. The matched graph is available in ``self.graph``
        """
        # Create a defaultdict to track number of matches per treatment ID
        preliminary_matches: Dict[str, List[str]] = defaultdict(list)
        seen_cids = set()

        # Sort edges by weight, increasing
        sorted_edges = sorted((uve for uve in self.graph.edges(data="weight")), key=lambda tup: tup[2])  # type: ignore
        for u, v, *_ in sorted_edges:
            u_attr = self.graph.nodes[u]["bipartite"]
            v_attr = self.graph.nodes[v]["bipartite"]

            # Identify treatment and control
            if u_attr == self._params.treatment_bipartite_attr and v_attr == self._params.control_bipartite_attr:
                tid, cid = u, v
            elif v_attr == self._params.treatment_bipartite_attr and u_attr == self._params.control_bipartite_attr:
                tid, cid = v, u
            else:
                raise NotBipartiteError

            if len(preliminary_matches[tid]) < n_match and (replace or cid not in seen_cids):
                preliminary_matches[tid].append(cid)
                seen_cids.add(cid)

        # Drop keys with less than ``min_match`` matches
        greedy_matches = {tid: cids for tid, cids in preliminary_matches.items() if len(cids) >= min_match}
        return self._filter_graph_by_match_dict(greedy_matches)

    def draw(self, with_labels: bool = False, *args, **kwargs) -> None:
        """Draw the graph.
        You can adjust the colors using ``self._params.treatment_color`` and ``self._params.control_color``

        Parameters
        ----------
        with_labels: bool
            Whether to draw the ``ids`` on the graph
        *args, **kwargs
            Passed to ``networkx.draw_networkx``
        """
        # BUG: kamada kawai layout is bugged?? Sometimes it doesn't end up drawing all the nodes...
        # NOTE: Just going to use default ``networkx.draw_networkx`` layout for now
        # pos = nx.kamada_kawai_layout(self.graph)

        nx.draw_networkx(self.graph, node_color=self.node_colors, with_labels=with_labels, *args, **kwargs)

    def draw_bipartite(self, with_labels: bool = False, *args, **kwargs) -> None:
        """Draw the graph using a bipartite layout, with the two groups clearly separated.
        You can adjust the colors using ``self._params.treatment_color`` and ``self._params.control_color``

        Parameters
        ----------
        with_labels: bool
            Whether to draw the ``ids`` on the graph
        *args, **kwargs
            Passed to ``networkx.draw_networkx``
        """
        top = self.graph_data.treatment.ids
        pos = nx.bipartite_layout(self.graph, nodes=top)
        nx.draw_networkx(self.graph, pos=pos, node_color=self.node_colors, with_labels=with_labels, *args, **kwargs)
