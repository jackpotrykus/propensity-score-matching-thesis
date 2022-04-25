from copy import deepcopy
from typing import List

import networkx as nx


def get_connected_subgraphs(G: nx.Graph, *, copy: bool = False, deep: bool = False, **kwargs) -> List[nx.Graph]:
    """Split the contents of `G` into a list of connected subgraphs.

    Parameters
    ----------
    copy: bool
        Whether or not to make a copy of `G` before computing the subgraphs. `False` by default
    deep: Optional[bool]
        Whether or not to make a _deep_ copy of `G`. Only used if `copy == True`. `False` by default.
    **kwargs
        Passed to `nx.Graph.copy()` if `copy=True` and `deep=False`

    Returns
    -------
    List[nx.Graph]
        Subgraphs of `G`, in a list. Each subgraph is fully connected
    """
    if copy:
        G2 = deepcopy(G) if deep else G.copy(**kwargs)
    else:
        G2 = G

    assert isinstance(G2, nx.Graph), "Copy failed"
    return [G2.subgraph(c) for c in nx.connected_components(G2)]
