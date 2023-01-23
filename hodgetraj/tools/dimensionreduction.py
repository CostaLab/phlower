import umap.umap_ as umap
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

def run_umap(mtx, random_state=2022):
    reducer = umap.UMAP(random_state=random_state)
    scaled_data = StandardScaler().fit_transform(mtx)
    embedding = reducer.fit_transform(scaled_data)
    return embedding

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

