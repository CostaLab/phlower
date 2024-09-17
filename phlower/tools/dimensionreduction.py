import umap.umap_ as umap
import scipy
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.spatial
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial import distance_matrix
from scipy.sparse import issparse, find,csr_matrix
from scipy.sparse.linalg import eigs
from .diffusionmap import affinity

def run_umap(mtx, random_state=2022):
    reducer = umap.UMAP(random_state=random_state)
    scaled_data = StandardScaler().fit_transform(mtx)
    embedding = reducer.fit_transform(scaled_data)
    return embedding

def run_tsne(mtx, n_components=2, random_state=2022, *args, **kwargs):
    tsne = TSNE(n_components=n_components, random_state=random_state, *args, **kwargs)
    dm = tsne.fit_transform(mtx)
    return dm

def run_isomap(mtx, n_components=2, n_neighbors=5, *args, **kwargs):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, *args, **kwargs)
    dm = isomap.fit_transform(mtx)
    return dm

def run_lda(mtx, y, n_components=2, random_state=2022, *args, **kwargs):
    lda = LinearDiscriminantAnalysis(n_components=n_components, random_state=random_state, *args, **kwargs)
    dm = lda.fit_transform(mtx, y)
    return dm

def run_pca(mtx, n_components=2, random_state=2022):
    dm = None
    if scipy.sparse.issparse(mtx):
        clf = TruncatedSVD(n_components, random_state=random_state)
        dm = clf.fit_transform(mtx)
        pass
    else:
        pca = PCA(n_components=n_components, random_state=random_state)
        dm = pca.fit_transform(mtx)
    return dm

def run_mds(mtx, n_components=2, random_state=2022, *args, **kwargs):
    mds = MDS(n_components=n_components, random_state=random_state, *args, **kwargs)
    dm = mds.fit_transform(mtx)
    return dm

def run_kernelpca(mtx, n_components=2, kernel='linear', random_state=2022, *args, **kwargs):
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel,random_state=random_state)
    dm = kpca.fit_transform(mtx)
    return dm


def run_palantir_diffusion_maps(data_df, n_components=10, knn=30, alpha=0, seed=None):
    """Run Diffusion maps using the adaptive anisotropic kernel

    :param data_df: PCA projections of the data or adjacency matrix
    :param n_components: Number of diffusion components
    :param knn: Number of nearest neighbors for graph construction
    :param alpha: Normalization parameter for the diffusion operator
    :param seed: Numpy random seed, randomized if None, set to an arbitrary integer for reproducibility
    :return: Diffusion components, corresponding eigen values and the diffusion operator
    """

    # Determine the kernel
    N = data_df.shape[0]
    if not issparse(data_df):
        print("Determing nearest neighbor graph...")
        temp = sc.AnnData(data_df.values)
        sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
        kNN = temp.obsp['distances']

        # Adaptive k
        adaptive_k = int(np.floor(knn / 3))
        adaptive_std = np.zeros(N)

        for i in np.arange(len(adaptive_std)):
            adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
                adaptive_k - 1
            ]

        # Kernel
        x, y, dists = find(kNN)

        # X, y specific stds
        dists = dists / adaptive_std[x]
        W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

        # Diffusion components
        kernel = W + W.T
    else:
        kernel = data_df

    # Markov
    D = np.ravel(kernel.sum(axis=1))

    if alpha > 0:
        # L_alpha
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)
        D = np.ravel(kernel.sum(axis=1))

    D[D != 0] = 1 / D[D != 0]
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
    # Eigen value dcomposition
    np.random.seed(seed)
    v0 = np.random.rand(min(T.shape))
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000, v0=v0)
    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    # Normalize
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    # Create are results dictionary
    res = {"T": T, "EigenVectors": V, "EigenValues": D}
    res["EigenVectors"] = pd.DataFrame(res["EigenVectors"])
    if not issparse(data_df):
        res["EigenVectors"].index = data_df.index
    res["EigenValues"] = pd.Series(res["EigenValues"])
    res["kernel"] = kernel

    return res

def run_palantir_fdl(mtx,
                     cell_names=None,
                     verbose=True,
                     iterations=500,
                     device='cpu',
                     knn=30,
                     alpha=0,
                     edgeWeightInfluence=0.5,
                     scalingRatio=2.0,
                     gravity = 1.0,
                     random_state=2022,
                     n_components=10,
                     outboundAttractionDistribution=False,
                     strongGravityMode=False):

    """" Function to compute force directed layout from the affinity_matrix
    adjusted from: https://github.com/dpeerlab/Harmony/blob/master/src/harmony/plot.py
    :param affinity_matrix: Sparse matrix representing affinities between cells
    :param cell_names: pandas Series object with cell names
    :param verbose: Verbosity for force directed layout computation
    :param iterations: Number of iterations used by ForceAtlas
    :return: Pandas data frame representing the force directed layout
    """
    from fa2 import ForceAtlas2
    if verbose:
        print(datetime.now(), "Computing palantir diffusion...")
    if type(mtx) != pd.DataFrame:
        mtx = pd.DataFrame(mtx)
    ## only affinity matrix is needed, so the n_components can be anything.
    affinity_matrix = run_palantir_diffusion_maps(mtx, n_components=n_components, knn=knn, alpha=alpha, seed=random_state)['kernel']

    init_coords = np.random.random((affinity_matrix.shape[0], 2))
    if verbose:
        print(datetime.now(), "Running force directed layout...")
    if device == 'cpu':
        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=outboundAttractionDistribution,  # Dissuade hubs
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=edgeWeightInfluence,
            # Performance
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,
            # Tuning
            scalingRatio=scalingRatio,
            strongGravityMode=strongGravityMode,
            gravity=gravity,
            # Log
            verbose=verbose)

        positions = forceatlas2.forceatlas2(
            affinity_matrix, pos=init_coords, iterations=iterations)
        positions = np.array(positions)

    elif device == 'gpu':
        import cugraph
        import cudf
        offsets = cudf.Series(affinity_matrix.indptr)
        indices = cudf.Series(affinity_matrix.indices)
        G = cugraph.Graph()
        G.from_cudf_adjlist(offsets, indices, None)

        forceatlas2 = cugraph.layout.force_atlas2(
            G,
            max_iter=iterations,
            pos_list=cudf.DataFrame(
                {
                    "vertex": np.arange(init_coords.shape[0]),
                    "x": init_coords[:, 0],
                    "y": init_coords[:, 1],
                }
            ),
            outbound_attraction_distribution=outboundAttractionDistribution,
            lin_log_mode=False,
            edge_weight_influence=edgeWeightInfluence,
            jitter_tolerance=1.0,
            barnes_hut_optimize=True,
            barnes_hut_theta=1.2,
            scaling_ratio=scalingRatio,
            strong_gravity_mode=strongGravityMode,
            gravity=gravity,
            verbose=True,
        )
        positions = forceatlas2.to_pandas().loc[:, ["x", "y"]].values

    # Convert to dataframe
    if cell_names is None:
        cell_names = np.arange(affinity_matrix.shape[0])

   # positions = pd.DataFrame(positions,
   #                          index=cell_names, columns=['x', 'y'])
    return positions


def outlier_removal(adata, umap='umap', bandwidth=.75, percentile=1, verbose=True):
    """
    remove umap outliers using kde for triangulation.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    umap : str
        umap key in adata.obsm
    bandwidth : float
        bandwidth for kde
    percentile : float
        percentile for outlier removal

    """
    import numpy as np
# import seaborn as sns # you probably can use seaborn to get pdf-estimation values, I would use scikit-learn package for this.
    from matplotlib import pyplot as plt
    from sklearn.neighbors import KernelDensity

    if umap not in adata.obsm.keys():
        raise ValueError("umap key not found in adata.obsm")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100")

    data = np.array(adata.obsm[umap])

    # you can use kernel='gaussian' instead
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth).fit(data)

    yvals = kde.score_samples(data)  # yvals are logs of pdf-values
    yvals[np.isinf(yvals)] = np.nan # some values are -inf, set them to nan

    # approx. 10 percent of smallest pdf-values: lets treat them as outliers
    outlier_inds = np.where(yvals < np.percentile(yvals, percentile))[0]
    #print(outlier_inds)
    non_outlier_inds = np.where(yvals >= np.percentile(yvals, percentile))[0]

    if verbose:
        print(f"Removing {len(outlier_inds)} outliers")

    adata = adata[non_outlier_inds, :].copy()
    return adata

def outlier_removal_clusters(adata,
                             cluster_slot='group',
                             umap='umap',
                             bandwidth=.75,
                             percentile=1,
                             kernel='tophat',
                             exclude_clusters=[],
                             verbose=True):
    """
    remove umap outliers using kde for triangulation clusterwise.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_slot : str
        Slot name for the cluster labels.
    umap : str
        Slot name for the umap coordinates or any 2D embeddings.
    bandwidth : float
        Bandwidth for the kernel density estimation.
    percentile : float
        Percentile for the outlier detection.
    kernel : str
        Kernel for the kernel density estimation.
    verbose : bool
        Print progress to stdout.
    """
    import numpy as np
# import seaborn as sns # you probably can use seaborn to get pdf-estimation values, I would use scikit-learn package for this.
    from matplotlib import pyplot as plt
    from sklearn.neighbors import KernelDensity
    from datetime import datetime


    if cluster_slot not in adata.obs:
        raise ValueError(f"adata.obs['{cluster_slot}'] does not exist.")
    if set(exclude_clusters) - set(adata.obs[cluster_slot]):
        raise ValueError(f"exclude_clusters contains unknown clusters.")
    if umap not in adata.obsm:
        raise ValueError(f"adata.obsm['{umap}'] does not exist.")
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")


    cell_list = []
    for c in set(adata.obs[cluster_slot]) - set(exclude_clusters):

        suba = adata[adata.obs[cluster_slot] ==c,:]

        data = np.array(suba.obsm[umap])
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)

        yvals = kde.score_samples(data)
        yvals[np.isinf(yvals)] = np.nan

        outlier_inds = np.where(yvals < np.percentile(yvals, percentile))[0]
        non_outlier_inds = np.where(yvals >= np.percentile(yvals, percentile))[0]

        cells = list(suba[non_outlier_inds, :].obs_names)
        cell_list.extend(cells)
        if verbose:
            print(datetime.now(), 'cleaned', c, len(outlier_inds),flush=True)
    for c in exclude_clusters: ## add back all cells
        cells = list(adata[adata.obs[cluster_slot] ==c, :].obs_names)
        cell_list.extend(cells)

    adata =  adata[cell_list, :].copy()
    #n1 = adata.n_obs
    #adata = outlier_removal(adata, umap=umap, bandwidth=bandwidth, percentile=percentile)
    #n2 = adata.n_obs
    #if verbose:
    #    print(datetime.now(), 'last cleaned', n1-n2,flush=True)


    return adata
#endf outlier_removal_clusters


def run_fdl(mtx,
            affinity_k = 7,
            #n_components=2,
            random_state=2022,
            outboundAttractionDistribution=False,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            # Performance
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,
            # Tuning
            scalingRatio=4.0,#2.0,
            strongGravityMode=True, #False,
            gravity=2.0, #1.0
            device = "cpu",
            # Log
            verbose=True,
            *args, **kwargs):
    """
    adjusted from: https://github.com/dpeerlab/Harmony/blob/master/src/harmony/plot.py
    ForceAtlas2
    input dm to calculate affinity matrix
    """
    from fa2 import ForceAtlas2


    np.random.seed(random_state)
    if verbose:
        print(datetime.now(), "Computing distance matrix...")
    R = distance_matrix(mtx, mtx)
    if verbose:
        print(datetime.now(), "Computing affinity matrix...")
    affinity_mtx = affinity(R, k=affinity_k,log=False, normalize=False)

    if verbose:
        print(datetime.now(), "Computing force directed layout...")
    init_coords = np.random.random((affinity_mtx.shape[0], 2))
    if device == "cpu":
        forceatlas2 = ForceAtlas2(
                    # Behavior alternatives
                    outboundAttractionDistribution=outboundAttractionDistribution,  # Dissuade hubs
                    linLogMode=linLogMode,
                    adjustSizes=adjustSizes,  # Prevent overlap (NOT IMPLEMENTED)
                    edgeWeightInfluence=edgeWeightInfluence,
                    # Performance
                    jitterTolerance=jitterTolerance,  # Tolerance
                    barnesHutOptimize=barnesHutOptimize,
                    barnesHutTheta=barnesHutTheta,
                    multiThreaded=multiThreaded,
                    # Tuning
                    scalingRatio=scalingRatio,
                    strongGravityMode=strongGravityMode,
                    gravity=gravity,
                    # Log
                    verbose=verbose,
                    *args,
                    **kwargs)

        positions = forceatlas2.forceatlas2(affinity_mtx, pos=init_coords, iterations=500)

    elif device == "gpu":
        import cugraph
        import cudf
        offsets = cudf.Series(affinity_mtx.indptr)
        indices = cudf.Series(affinity_mtx.indices)
        G = cugraph.Graph()
        G.from_cudf_adjlist(offsets, indices, None)
        forceatlas2 = cugraph.layout.force_atlas2(
            G,
            max_iter=iterations,
            pos_list=cudf.DataFrame(
                {
                    "vertex": np.arange(init_coords.shape[0]),
                    "x": init_coords[:, 0],
                    "y": init_coords[:, 1],
                }
            ),
            outbound_attraction_distribution=outboundAttractionDistribution,
            lin_log_mode=linLogMode,
            edge_weight_influence=edgeWeightInfluence,
            jitter_tolerance=jitterTolerance,
            barnes_hut_optimize=barnesHutOptimize,
            barnes_hut_theta=barnesHutTheta,
            scaling_ratio=scalingRatio,
            strong_gravity_mode=strongGravityMode,
            gravity=gravity,
            verbose=verbose,
            *args,
            **kwargs,
        )

        positions = forceatlas2.to_pandas().loc[:, ["x", "y"]].values

    positions = np.array(positions)
    return positions
