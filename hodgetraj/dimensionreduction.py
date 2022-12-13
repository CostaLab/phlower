import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_umap(mtx, random_state=2022):
    reducer = umap.UMAP(random_state=random_state)
    scaled_data = StandardScaler().fit_transform(mtx)
    embedding = reducer.fit_transform(scaled_data)
    return embedding

def run_pca(mtx, n_components=2):
    pca = PCA(n_components=n_components)
    dm = pca.fit_transform(mtx)
    return dm

