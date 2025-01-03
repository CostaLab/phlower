import igraph as ig
import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

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

def agglomerativeclustering(embedding, distance='euclidean', n_clusters=2, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None):
    from sklearn.cluster import AgglomerativeClustering
    ac = AgglomerativeClustering(n_clusters=n_clusters, metric=affinity, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage, distance_threshold=distance_threshold)
    return ac.fit_predict(embedding)

def spectralclustering(embedding, distance_matrix=None, distance='euclidean', precompute=False, n_clusters=8, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None):
    from sklearn.cluster import SpectralClustering

    _distances = None
    if precompute and distance_matrix is not None:
        _distances = distance_matrix
    elif not precompute and embedding is not None:
        print("calcaulating distances...")
        _distances = pairwise_distances(embedding, metric=distance)
    else:
        raise ValueError("Either embedding or distance_matrix must be provided")

    print("spectral clustering...")
    sc = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, n_components=n_components, random_state=random_state, n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, degree=degree, coef0=coef0, kernel_params=kernel_params, n_jobs=n_jobs)
    return sc.fit_predict(_distances)


def gaussianmixture(embedding, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                         tol=tol, reg_covar=reg_covar, max_iter=max_iter,
                         n_init=n_init, init_params=init_params, weights_init=weights_init,
                         means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start=warm_start,
                         verbose=verbose, verbose_interval=verbose_interval)
    gm.fit(embedding)
    return gm.predict(embedding)

def leiden(embedding=None, distance_matrix=None, distance='euclidean', precompute=False, n_iterations=-1, resolution=1.0,  seed_state=2022, **partition_kwargs):
    import leidenalg as la

    _distances = None
    if precompute and distance_matrix is not None:
        _distances = distance_matrix
    elif not precompute and embedding is not None:
        print("calcaulating distances...")
        _distances = pairwise_distances(embedding, metric=distance)
    else:
        raise ValueError("Either embedding or distance_matrix must be provided")
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



def louvain(embedding, distance_matrix=None, distance='euclidean', precompute=False, resolution=1.0,  seed_state=2022, **partition_kwargs):
    import louvain as lv

    _distances = None
    if precompute and distance_matrix is not None:
        _distances = distance_matrix
    elif not precompute and embedding is not None:
        print("calcaulating distances...")
        _distances = pairwise_distances(embedding, metric=distance)
    else:
        raise ValueError("Either embedding or distance_matrix must be provided")

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




def meta_cells_adata(
        adata,
        embedding_key='X_pca',
        embedding_comps=None,
        n_clusters = None,
        resolution = 30,
        n_comps = 30,
        seed = 2022,
        flavor = "hier",
        ):
    """\
    adjust from: https://github.com/PeterZZQ/CellPath.git
    Cluster cells into clusters, using K-means

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_clusters
        number of clusters, default, cell number/10

    Returns
    -------
    adata if copy=True else None
    """
    if n_clusters == None:
        n_clusters = int(adata.n_obs/10)

    kmeans = KMeans(n_clusters = n_clusters, init = "k-means++", n_init = 10, max_iter = 300, tol = 0.0001, random_state = seed)

    X_pca =adata.obsm[embedding_key][:, 0:adata.obsm[embedding_key].shape[1] if  n_comps is None else n_comps]
    if flavor == "k-means":
        print("using k-means", flush=True)
        metacells = kmeans.fit_predict(X_pca)
    elif flavor == "leiden":
        print("using leiden", flush=True)
        metacells = leiden(X_pca, resolution = resolution, seed_state = seed)
    elif flavor == "hier":
        print("using hier", flush=True)
        metacells = AgglomerativeClustering(n_clusters = n_clusters, metric= "euclidean").fit(X_pca).labels_
    else:
        raise ValueError("flavor can only be `k-means', `leiden' or `hier'")

    adata.obs['metacells'] = metacells.astype('int64')

    #badata = adata.copy()
    X = np.zeros((adata.n_vars, n_clusters))
    for i in range(n_clusters):
        X[:, i] = adata[adata.obs['metacells'] == i, :].X.mean(0)

    bdata = sc.AnnData(X, obs = adata.var)


    for i in range(n_clusters):
        for a_slot in adata.obs.columns:
            if type(adata.obs[a_slot][0]) == str: ## find the majority
                bdata.obs[a_slot][adata.obs['metacells'] == i] = Counter(adata.obs[a_slot][adata.obs['metacells'] == i]).most_common(1)[0][0]
            elif type(adata.obs[a_slot][0]) == np.int64 or type(adata.obs[a_slot][0]) == np.float64: ## mean
                bdata.obs[a_slot][adata.obs['metacells'] == i] = np.mean(adata.obs[a_slot][adata.obs['metacells'] == i])

    ## add metacell coordinates
    metacell_coords = np.zeros((n_clusters, X_pca.shape[1]))
    for i in range(n_clusters):
        metacell_coords[i,:] = np.mean(X_pca[metacells == i,:], axis = 0)
    bdata.uns[embedding_key] = metacell_coords

    return bdata

