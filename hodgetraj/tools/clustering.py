import igraph as ig
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN

def _get_igraph_from_adjacency(adjacency, directed=None):
    """
    adjust from https://github.com/scverse/scanpy.git
    Get igraph graph from adjacency matrix.
    """

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        print(f'Warning: The constructed graph has only {g.vcount()} nodes. '
               'Your adjacency matrix contained redundant nodes.'
        )
    return g


def leiden(embedding, distance='euclidean',n_iterations=-1, resolution=1.0,  seed_state=0, **partition_kwargs):
    import leidenalg as la

    print("calcaulating distances...")
    _distances = pairwise_distances(embedding, metric=distance)
    connectivities = 1/(1+_distances)

    gclust = _get_igraph_from_adjacency(connectivities, directed=False)
    partition_type = la.RBConfigurationVertexPartition
    partition_kwargs['weights'] = np.array(gclust.es['weight']).astype(np.float64)
    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = seed_state
    if resolution is not None:
        partition_kwargs['resolution_parameter'] = resolution

    print("leiden clustering...")
    part = la.find_partition(gclust, partition_type, **partition_kwargs)
    groups = np.array(part.membership)
    return groups



def louvain(embedding, distance='euclidean', resolution=1.0,  seed_state=0, **partition_kwargs):
    import louvain as lv

    print("calcaulating distances...")
    _distances = pairwise_distances(embedding, metric=distance)
    connectivities = 1/(1+_distances)

    gclust = _get_igraph_from_adjacency(connectivities, directed=False)
    partition_type = lv.RBConfigurationVertexPartition
    partition_kwargs['weights'] = np.array(gclust.es['weight']).astype(np.float64)
    partition_kwargs['seed'] = seed_state
    if resolution is not None:
        partition_kwargs['resolution_parameter'] = resolution

    print("louvain clustering...")
    part = lv.find_partition(
                gclust,
                partition_type,
                **partition_kwargs,
            )
    groups = np.array(part.membership)
    return groups


def dbscan(embedding, distance='euclidean',  eps=0.5,min_samples=5,metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
    clustering = DBSCAN(eps=eps,
                        min_samples=min_samples,
                        metric=distance,
                        metric_params=metric_params,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        p=p,
                        n_jobs=n_jobs).fit(embedding)
    return clustering.labels_