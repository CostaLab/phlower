import numpy as np
import networkx as nx
from anndata import AnnData
from sklearn.neighbors import KDTree
from tqdm import trange
from collections import Counter, defaultdict
from ..util import bsplit
from .trajectory import M_create_matrix_coordinates_trajectory_Hspace


def num_times(a,b):
    if a == 0 or b == 0:
        return np.nan
    return a/b if a>b else b/a

def htraj_matrix(adata=None,
                 evector_name="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                 full_traj_matrix = "full_traj_matrix",
                 trajs_clusters = 'trajs_clusters',
                 trajs_use = 100,
                 eigen_n = 2,
                ):

    """
    Create a matrix:
        each column is the harmonic dimension of eigen_n,
        each row is a trajectory edge.
        there are average(len(a_traj))*trajs_num rows

    Parameters
    ----------
    trajs_use: number of trajectories to use to create the Hspace matrix
    eigen_n: number of eigen vectors to use to calculate the KNN


    Returns
    -------
    mat: matrix of the htraj_matrix
    row_ids: list of a tuple(trajectory_id, edge_idx)
    row_clusters: assign the cluster id for each trajectory edge point
    row_edges: assign the edge id for each trajectory edge point
    dic_traj_starts_idx: assign the start index for each trajectory
    cumsums: assign the Hspace cumsum of the trajectory edge points for each trajectory
    """
    m_full_traj_matrix = adata.uns[full_traj_matrix]
    trajs_use = min(trajs_use, len(m_full_traj_matrix))
    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:eigen_n, :], adata.uns[full_traj_matrix][0:trajs_use])
    cumsums = list(map(lambda i: [np.cumsum(j) for j in i ], mat_coord_Hspace))

    mat = None
    row_ids = np.array([], dtype=int)
    row_clusters = []
    row_edges = [] ## store trajectory edges info
    dic_traj_starts_idx = {0:0} #store where to start for each trajectory
    for itraj in trange(len(cumsums)):
        nmtx = np.vstack(cumsums[itraj]).T
        traj_mtx = adata.uns[full_traj_matrix][itraj]
        traj_edge_idx = [j for i in np.argmax(np.abs(traj_mtx.astype(int)), axis=0).tolist() for j in i]

        dic_traj_starts_idx[itraj] = len(row_edges)
        if mat is None:
            mat = nmtx
            row_ids = np.array(list(zip([itraj]*nmtx.shape[0], range(nmtx.shape[0]))), dtype=int)
            row_edges = traj_edge_idx
        else:
            mat = np.concatenate((mat, nmtx))
            row_ids = np.concatenate((row_ids, np.array(list(zip([itraj]*nmtx.shape[0], range(nmtx.shape[0]))))))
            row_edges.extend(traj_edge_idx)

        row_clusters.extend([adata.uns[trajs_clusters][itraj]]*nmtx.shape[0])

    return mat, row_ids, np.array(row_clusters), row_edges, dic_traj_starts_idx, cumsums


def traj_knn(coor_mat, k=100, **args):
    tree = KDTree(coor_mat, **args)
    distances, indices = tree.query(coor_mat, k)
    return distances, indices


def harmonic_tree(adata: AnnData,
                  evector_name="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                  full_traj_matrix = "full_traj_matrix",
                  trajs_clusters = 'annotation',
                  k = 100,
                  eigen_n = 20,
                  trajs_use = 10000,
                  tree_name = 'htree',
                  ):

    """
    Parameters
    ----------
    k: number of nearest neighbors for travelling the harmonic space
    """
    print("Creating Hspace matrix...")
    mat, row_ids, row_clusters, row_edges, dic_traj_starts_idx, cumsums = htraj_matrix(adata,
                                                                                       evector_name=evector_name,
                                                                                       full_traj_matrix = full_traj_matrix,
                                                                                       eigen_n=eigen_n,
                                                                                       trajs_use = trajs_use,
                                                                                       trajs_clusters=trajs_clusters)


    print("calculating k nearest neighbors...")
    distances, indices = traj_knn(mat, k=k)

    print("counting the trajectory groups...")
    keys_counters = defaultdict(int)
    for i in trange(0, min(trajs_use, len(cumsums))):
        dic = atraj_edges_split(cumsums[i], i, row_clusters, indices, dic_traj_starts_idx, isprint=False)
        for key in dic:
            keys_counters[key] += dic[key]

    full_list = list(set(adata.uns[trajs_clusters]))
    tree_list=[]
    htree = nx.DiGraph()
    create_harmonic_tree_list(full_list, tree_list, htree, keys_counters)
    return htree



def atraj_edges_split(cumsum, itraj, row_clusters, indices, dic_traj_starts_idx, isprint=False):

    dic_keys_counts = defaultdict(int)
    if not isinstance(row_clusters, np.ndarray):
        row_clusters = np.array(row_clusters)
    for i in range(cumsum[0].shape[0]):
        nn = (row_clusters[indices[i+dic_traj_starts_idx[itraj]]])
        dd = Counter(nn)
        for k,v in list(dd.items()):
            if v <3:
                del dd[k]
        dic_keys_counts[tuple(set(dd.keys()))] += 1

        if isprint:
            print(i, len(dd.keys()) ,dd)
    return dic_keys_counts




def create_harmonic_tree_list(lst, tree_list, htree, keys_counters, times_diff=50):
    ddd = defaultdict(int)
    if len(lst) <2:
        return
    split_list =  bsplit(lst)
    for k1,k2 in split_list:
        if keys_counters.get(k1,0) > 0 and keys_counters.get(k2,0) >0:
            if num_times(keys_counters.get(k1,0), keys_counters.get(k2,0)) > times_diff: ## !!! here probably no output for any, should return the best
                continue
            if (k2, k1) in keys_counters:
                continue
            ddd[(k1,k2)] = keys_counters.get(k1,0) + keys_counters.get(k2,0)
    if len(ddd.items()) > 0:
        lst1, lst2 = sorted(ddd.items(), key=lambda x:x[1], reverse=True)[0][0]
        #print(f"add {lst}->{lst1}                  {lst}->{lst2}")
        htree.add_edge(tuple(lst), lst1)
        htree.add_edge(tuple(lst), lst2)
        tree_list.append((lst1, lst2))
        create_harmonic_tree_list(lst1, tree_list, htree, keys_counters)
        create_harmonic_tree_list(lst2, tree_list, htree, keys_counters)
    else:
        return



