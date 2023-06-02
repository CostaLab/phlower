import random
import networkx as nx
import numpy as np
from tqdm import trange
from anndata import AnnData
from typing import List
from collections import Counter, defaultdict
from itertools import chain
from scipy.sparse import csr_matrix
from typing import Union

from .graphconstr import adjedges, edges_on_path
from .dimensionreduction import run_umap, run_pca
from .clustering import dbscan, leiden, louvain
from .trajectory import M_create_matrix_coordinates_trajectory_Hspace
from ..util import pairwise, find_knee, tuple_increase, pearsonr_2D


def feature_mat_coor_flatten_trajectory(adata: AnnData,
                                        feature : str = None,
                                        graph_name: str = None,
                                        evector_name: str = None,
                                        full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                                        dims = [0,1],
                                        ):

    if not feature in adata.var_names:
        raise ValueError(f'Feature {feature} not in adata.var_names')

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    featureidx = np.where(adata.var_names == feature)[0]
    edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidx] for y in x])
    traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
    # speed up
    #mat_coor_flatten_trajectory = [np.einsum("ij, jk -> ik", adata.uns[evector_name][0:max(dims)+1, :],  mat) for mat in traj_score] #most time consuming
    mat_coor_flatten_trajectory = (adata.uns[evector_name][0:max(dims)+1, :] @ traj_score.T).T #most time consuming
    return mat_coor_flatten_trajectory



def feature_mat_coor_flatten_trajectory_direction(adata: AnnData,
                                                  feature : str = None,
                                                  graph_name: str = None,
                                                  evector_name: str = None,
                                                  full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                                                  u_attribute = 'u',
                                                  dims = [0,1],
                                                  ):

    if not feature in adata.var_names:
        raise ValueError(f'Feature {feature} not in adata.var_names')

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    featureidx = np.where(adata.var_names == feature)[0]

    edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidx] for y in x])

    elist = adata.uns[graph_name].edges()
    elist_set = set(list(elist))

    if u_attribute is not None:
        u=nx.get_node_attributes(adata.uns[graph_name], u_attribute)
        direction = [-1 if (u[e]-u[a])>0 else 1 for a,e in elist]
        edges_score= np.array([i*j for i,j in zip(edges_score, direction)])

    traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
    # speed up
    #mat_coor_flatten_trajectory = [np.einsum("ij, jk -> ik", adata.uns[evector_name][0:max(dims)+1, :],  mat) for mat in traj_score] #most time consuming
    mat_coor_flatten_trajectory = (adata.uns[evector_name][0:max(dims)+1, :] @ traj_score.T).T #most time consuming
    return mat_coor_flatten_trajectory



def feature_cumsum_direction(adata: AnnData,
                             feature : str = None,
                             graph_name: str = None,
                             evector_name: str = None,
                             full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                             full_traj_matrix: str = 'full_traj_matrix',
                             u_attribute = 'u',
                             eigen_n = -1,
                             dims = [0,1],
                             ):


    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"



    if eigen_n < 1:
        print("eigen_n < 1, use knee_eigen to find the number of eigen vectors to use")
        if "eigen_value_knee" in adata.uns.keys():
            eigen_n = adata.uns["eigen_value_knee"]
        else:
            eigen_n = knee_eigen(adata, eigens=re.sub(r"_vector$", r"_value", evector_name) , plot=False)



    m_full_traj_matrix = adata.uns[full_traj_matrix]
    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:eigen_n, :], adata.uns[full_traj_matrix])
    cumsums = list(map(lambda i: [np.cumsum(j) for j in i ], mat_coord_Hspace))
    cumsums = np.array(cumsums)

    elist = adata.uns[graph_name].edges()
    elist_set = set(list(elist))

    featureidx = np.where(adata.var_names == feature)[0]
    edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidx] for y in x])
    if u_attribute is not None:
        u=nx.get_node_attributes(adata.uns[graph_name], u_attribute)
        direction = [-1 if (u[e]-u[a])>0 else 1 for a,e in elist]
        edges_score= np.array([i*j for i,j in zip(edges_score, direction)])

    traj_score = np.multiply(cumsums, edges_score[None, :])

    mat_coor_flatten_trajectory = (adata.uns[evector_name][0:max(dims)+1, :] @ traj_score.T).T #most time consuming
    return mat_coord_flatten_trajectory



#endf feature_cumsum_direction


def get_featuresidx(adata: AnnData,
                    features: List[str] = None,
                    ):
    featuresidx = [np.where(adata.var_names == feature)[0][0] for feature in features]
    return featuresidx



def feature_harmonic_multiply(adata: AnnData,
                                features : Union[str, List[str]] = None,
                                graph_name: str = None,
                                evector_name : str = None,
                                full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                                trajs_clusters: str = 'trajs_clusters',
                                dims=list(range(0,10)),
                                st = np.std,
                                ):


    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    features = [features] if isinstance(features, str) else features
    features = [feature for feature in features if feature in adata.var_names]
    if len(features) == 0:
        raise ValueError(f'No features in adata.var_names')

    featureidxs = [np.where(adata.var_names == feature)[0] for feature in features]

    evec = adata.uns[evector_name][0:max(dims)+1, :]
    l_trajs_clusters =adata.uns[trajs_clusters]
    clusters = set(l_trajs_clusters)

    dics = {}
    for i in trange(len(features), desc='features correlation'):
        feature = features[i]
        edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidxs[i]] for y in x])
        traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
        scores = np.multiply(evec,  edges_score[None, :]).T
        dics[features[i]] = scores
    return dics


def feature_correlation_cluster(adata: AnnData,
                              features : Union[str, List[str]] = None,
                              graph_name: str = None,
                              evector_name : str = None,
                              full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                              trajs_clusters: str = 'trajs_clusters',
                              dim=0,
                              st = np.mean,
                              ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"


    features = [features] if isinstance(features, str) else features
    features = [feature for feature in features if feature in adata.var_names]
    if len(features) == 0:
        raise ValueError(f'No features in adata.var_names')

    featureidxs = [np.where(adata.var_names == feature)[0] for feature in features]

    evec = adata.uns[evector_name][dim, :]
    l_trajs_clusters =adata.uns[trajs_clusters]
    clusters = set(l_trajs_clusters)
    if not isinstance(l_trajs_clusters, np.ndarray):
        l_trajs_clusters = np.array(l_trajs_clusters, dtype=object)

    dics = {}
    for i in trange(len(features), desc='features statistic cluster'):
        feature = features[i]
        edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidxs[i]] for y in x])
        traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
        dic = {}
        #multiplied = evec @ adata.uns[full_traj_matrix_flatten].T
        for cluster in clusters:
            idx = np.where(l_trajs_clusters==cluster)[0]
            scores = np.mean(pearsonr_2D(evec, traj_score[idx, :]))
            dic[cluster] = scores
        dics[features[i]] = dic
    return dics



def feature_statistic_cluster(adata: AnnData,
                              features : Union[str, List[str]] = None,
                              graph_name: str = None,
                              evector_name : str = None,
                              full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                              trajs_clusters: str = 'trajs_clusters',
                              dims=list(range(0,10)),
                              st = np.std,
                              ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    features = [features] if isinstance(features, str) else features
    features = [feature for feature in features if feature in adata.var_names]
    if len(features) == 0:
        raise ValueError(f'No features in adata.var_names')

    featureidxs = [np.where(adata.var_names == feature)[0] for feature in features]

    evec = adata.uns[evector_name][0:max(dims)+1, :]
    l_trajs_clusters =adata.uns[trajs_clusters]
    clusters = set(l_trajs_clusters)
    if not isinstance(l_trajs_clusters, np.ndarray):
        l_trajs_clusters = np.array(l_trajs_clusters, dtype=object)


    dics = {}
    for i in trange(len(features), desc='features statistic cluster'):
        feature = features[i]
        edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidxs[i]] for y in x])
        traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
        dic = {}
        multiplied = evec @ traj_score.T
        for cluster in clusters:
            scores = st((multiplied).T[np.where(l_trajs_clusters==cluster)[0], :], axis=0)
            dic[cluster] = scores
        dics[features[i]] = dic
    return dics




def task_statistics(feature, evec, traj_score, l_trajs_clusters, clusters, st):
    dic = {}
    multiplied = evec @ traj_score.T
    if not isinstance(l_trajs_clusters, np.ndarray):
        l_trajs_clusters = np.array(l_trajs_clusters, dtype=object)

    for cluster in clusters:
        score = st((multiplied).T[np.where(l_trajs_clusters==cluster)[0], :])
        dic[cluster] = score
    return feature, dic

def feature_statistic_cluster2(adata: AnnData,
                              features : Union[str, List[str]] = None,
                              graph_name: str = None,
                              evector_name : str = None,
                              full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                              trajs_clusters: str = 'trajs_clusters',
                              dims=list(range(0,10)),
                              st = np.std,
                              n_jobs = 4,
                              ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    features = [features] if isinstance(features, str) else features
    features = [feature for feature in features if feature in adata.var_names]
    if len(features) == 0:
        raise ValueError(f'No features in adata.var_names')
    featureidxs = [np.where(adata.var_names == feature)[0] for feature in features]
    evec = adata.uns[evector_name][0:max(dims)+1, :]
    l_trajs_clusters =adata.uns[trajs_clusters]
    clusters = set(l_trajs_clusters)

    if not isinstance(l_trajs_clusters, np.ndarray):
        l_trajs_clusters = np.array(l_trajs_clusters, dtype=object)

    if n_jobs == 1:
        dics = {}
        for i in trange(len(features), desc='features statistic cluster'):
            feature = features[i]
            edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidxs[i]] for y in x])
            traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
            dic = {}
            for cluster in clusters:
                score = st((evec @ traj_score.T).T[np.where(l_trajs_clusters==cluster)[0], :])
                dic[cluster] = score
            dics[features[i]] = dic
        return dics
    else:
        #from multiprocessing import Pool
        from pathos.multiprocessing import ProcessingPool as Pool
        edges_scores = [G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidx] for y in x]) for featureidx, feature in zip(featureidxs, features)]
        traj_scores = [np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :]) for edges_score in edges_scores]
        with Pool(n_jobs) as p:
            return {k:v for k,v in  p.uimap(task_statistics, features, [evec]*len(features), traj_scores, [l_trajs_clusters]*len(features), [clusters]*len(features), [st]*len(features))}
            #r = p.uimap(task_statistics, features, [evec]*len(features), traj_scores, [l_trajs_clusters]*len(features), [clusters]*len(features), [st]*len(features))
        #return [x for x in r.get()]




def feature_statistic_cluster3(adata: AnnData,
                              features : Union[str, List[str]] = None,
                              graph_name: str = None,
                              evector_name : str = None,
                              full_traj_matrix_flatten: str = 'full_traj_matrix_flatten',
                              trajs_clusters: str = 'trajs_clusters',
                              dims=list(range(0,10)),
                              st = np.std,
                              n_jobs = 4,
                              ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    features = [features] if isinstance(features, str) else features
    features = [feature for feature in features if feature in adata.var_names]
    if len(features) == 0:
        raise ValueError(f'No features in adata.var_names')
    featureidxs = [np.where(adata.var_names == feature)[0] for feature in features]
    evec = adata.uns[evector_name][0:max(dims)+1, :]
    l_trajs_clusters =adata.uns[trajs_clusters]
    clusters = set(l_trajs_clusters)
    if not isinstance(l_trajs_clusters, np.ndarray):
        l_trajs_clusters = np.array(l_trajs_clusters, dtype=object)

    if n_jobs == 1:
        dics = {}
        for i in trange(len(features), desc='features statistic cluster'):
            feature = features[i]
            edges_score = G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidxs[i]] for y in x])
            traj_score = np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :])
            dic = {}
            for cluster in clusters:
                scores = st((evec @ traj_score.T).T[np.where(l_trajs_clusters==cluster)[0], :], axis=0)
                dic[cluster] = scores
            dics[features[i]] = dic
        return dics
    else:
        #from multiprocessing import Pool
        from pathos.multiprocessing import ProcessingPool as Pool
        from joblib import Parallel, delayed
        edges_scores = [G_features_edges(adata.uns[graph_name], [y for x in adata.X[:, featureidx] for y in x]) for featureidx, feature in zip(featureidxs, features)]
        traj_scores = [np.multiply(adata.uns[full_traj_matrix_flatten], edges_score[None, :]) for edges_score in edges_scores]
        return {k: v for k,v in zip(Parallel(n_jobs=n_jobs)(delayed(task_statistics)(feature, evec, traj_score, l_trajs_clusters, clusters, st) for feature, traj_score in zip(features, traj_scores)))}




def G_features_edges(G: nx.DiGraph,
                     feature_nodes=None,
                    ):
    A = np.subtract.outer(feature_nodes, feature_nodes)
    feature_edge_score = [A[i,j] for i,j in G.edges()]

    return np.array(feature_edge_score)


