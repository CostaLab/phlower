import umap.umap_ as umap
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from fa2 import ForceAtlas2

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



def run_fdl(mtx,
            affinity_k = 7,
            n_components=2,
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
            # Log
            verbose=True,
            *args, **kwargs):
    """
    ForceAtlas2
    input dm to calculate affinity matrix
    """



    np.random.seed(random_state)
    R = distance_matrix(mtx, mtx)
    affinity_mtx = phlower.tl.affinity(R, k=affinity_k,log=False, normalize=False)
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

    init_coords = np.random.random((affinity_mtx.shape[0], 2))
    positions = forceatlas2.forceatlas2(affinity_mtx, pos=init_coords, iterations=500)
    positions = np.array(positions)
    return positions

