##TODO
## remove noise clusters labelled by -1

import re
import random
import networkx as nx
import numpy as np
import pandas as pd
import tqdm
import itertools
import scipy
from datetime import datetime
from tqdm import trange
from anndata import AnnData
from typing import List
from collections import Counter, defaultdict
from itertools import chain
from scipy.sparse import csr_matrix
from typing import Union, List, Tuple
from sklearn.neighbors import NearestNeighbors
from .graphconstr import adjedges, edges_on_path
from .dimensionreduction import run_umap, run_pca
from .clustering import dbscan, leiden, louvain
from .hodgedecomp import knee_eigen
from ..util import pairwise, find_knee, tuple_increase, is_node_attr_existing
from .tree_utils import assign_graph_node_attr_to_adata


def random_climb_knn(adata,
                     graph_name = None,
                     A = None,
                     W = None,
                     knn_edges_k = 9,
                     attr:str='u',
                     roots_ratio:float=0.1,
                     n:int=10000,
                     iscopy=False,
                     traj_name = None,
                     knn_type= 'diffusion',
                     seeds:int=2022):
    """
    Randomly climb the graph from the roots cells to the leaves cells using the knn_edges.
    1. random climb the knn_edges graph from the root cells
    2. each edge in knn_edges, find the shortest path in the graph_name graph if there's no edge in the graph_name

    Parameters:
    ----------
    adata: AnnData
        AnnData object
    graph_name: str
        the graph with holes to be used, adata.uns["graph_basis"] + "_triangulation" by default if None
    A: csr_matrix
        the adjacency matrix of the diffusion
    W: csr_matrix
        the weight matrix of the diffusion
    knn_edges_k: int
        the number of knn edges to be used, 9 by default
    attr: str
        the attribute of the graph_name, "u" by default
    roots_ratio: float
        the ratio of the cells to be used as root, 0.1 by default
    n: int
        the number of trajectories to be produced, 10000 by default
    iscopy: bool
        whether to return a copy of adata or not, False by default
    traj_name: str
        the name of the trajectories to be saved in adata.uns, "knn_trajs" by default
    seeds: int
        the random seeds to be used, 2022 by default
    """

    adata = adata.copy() if iscopy else adata

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation"
        A = re.sub("_g$", "_A", adata.uns["graph_basis"])
        W = re.sub("_g$", "_W", adata.uns["graph_basis"])

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")
    if A not in adata.uns:
        raise ValueError(f"{A} not in adata.uns")
    if W not in adata.uns:
        raise ValueError(f"{W} not in adata.uns")
    if not is_node_attr_existing(adata.uns[graph_name], attr):
        raise ValueError(f"{attr} not in adata.uns[{graph_name}]")

    if n <=0:
        raise ValueError(f"{n}: number of trajectories should be > 0 ")
    if n <100:
        print(f"{n}: number of trajectories would be better when >= 100")

    g = adata.uns[graph_name]
    if knn_type =="diffusion":
        knn_edges = adjedges(adata.uns[A], adata.uns[W], knn_edges_k)
    elif knn_type =="euclidean":
        from sklearn.neighbors import kneighbors_graph
        basis = adata.uns['graph_basis']
        A = kneighbors_graph(adata.obsm[basis], 20, mode='connectivity', include_self=False)
        knn_edges = list(nx.from_numpy_matrix(A).edges())
    else:
        raise ValueError(f"{knn_type} not supported, only diffusion and euclidean are supported for now")
    knn_edges = [tuple_increase(i,j) for (i,j) in knn_edges]
    knn_trajs = G_random_climb_knn(g, knn_edges, attr=attr, roots_ratio=roots_ratio, n=n, seeds=seeds)
    if traj_name is None:
        traj_name = f"knn_trajs"

    adata.uns[traj_name] = knn_trajs
    adata.uns['climbing_root_ratio'] = roots_ratio

    return adata if iscopy else None


def trajs_matrix(adata: AnnData,
                 graph_name: str = None,
                 evector_name: str = None,
                 embedding = 'umap',
                 eigen_n: int = -1,
                 trajs : Union[str, List[List[int]]] = "knn_trajs",
                 edge_w : List = None,
                 iscopy = False,
                 verbose = True,
                ):
    """
    1. embed the trajectories to be the flows of SC
    2. embed the flow to be the harmonics space of the graph

    Parameters:
    ----------
    adata: AnnData
        AnnData object
    graph_name: str
        the graph with holes to be used, adata.uns["graph_basis"] + "_triangulation_circle" by default if None
    evector_name: str
        the L1 decomposed eigen vectors from the graph_name,  adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector" by default if None
    embedding: str
        the embedding for visualize the clustering results, "umap" by default
    eigen_n: int
        the number of eigen vectors to be used, -1 by default, which means all eigen vectors with 0 eigen values
    trajs: str or List[List[int]]
        the trajectories to be used, "knn_trajs" by default in adata.uns
    edge_w: List
        the weights of the edges in the graph, None by default, which means all edges have the same weight 1
    iscopy: bool
        whether to return a copy of adata or not, False by default
    """
    adata = adata.copy() if iscopy else adata

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"
    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")

    if verbose:
        print(datetime.now(), "projecting trajectories to simplics...")
    full_trajectory_matrix(adata,
                           trajs=trajs,
                           edge_w = edge_w,
                           iscopy = False)
    if verbose:
        print(datetime.now(), "Embedding trajectory harmonics...")
    trajs_dm(adata,
             evector_name = evector_name,
             M_flatten = "full_traj_matrix_flatten",
             embedding = embedding,
             eigen_n = eigen_n,
             iscopy = False
            )
    if verbose:
        print(datetime.now(), "done.")
    return adata if iscopy else None
#endf trajs_matrix

def full_trajectory_matrix(adata: AnnData,
                           graph_name: str = None,
                           trajs : Union[str, List[List[int]]] = "knn_trajs",
                           edge_w : List = None,
                           iscopy = False,
                           ):
    """
    We multiply the trajs matrix(nodes order) with the graph structure, 1 if same direction else -1 for each edge in a trajs.
    Otherwise set to be 0
    """
    adata = adata.copy() if iscopy else adata

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if graph_name not in adata.uns.keys():
        raise ValueError(f"{graph_name} not in adata.uns")

    if isinstance(trajs, str):
        if trajs not in adata.uns:
            raise ValueError(f"{trajs} not in adata.uns")
        trajs = adata.uns[trajs]

    g = adata.uns[graph_name]
    elist = np.array([(x[0], x[1]) for x in g.edges()])
    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}
    M_full = G_full_trajectory_matrix(g, map(lambda path: list(edges_on_path(path)), chain.from_iterable([trajs])), elist, elist_dict)
    adata.uns["full_traj_matrix"] = M_full
    adata.uns["full_traj_matrix_flatten"] = L_flatten_trajectory_matrix(M_full)
    adata.uns["full_traj_matrix_flatten_norm"] = L_flatten_trajectory_matrix_norm(M_full)


    return adata if iscopy else None


def trajs_dm(adata,
             evector_name: str = None,
             M_flatten: Union[str, np.ndarray] = "full_traj_matrix_flatten",
             embedding = 'umap',
             eigen_n: int = -1,
             iscopy=False
             ):
    """
    adata: AnnData
        AnnData object
    evector_name: str
        the L1 decomposed eigen vectors from the graph_name,  adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector" by default if None
    embedding: str
        the embedding for visualize the clustering results, "umap" by default
    eigen_n: int
        the number of eigen vectors to use, if -1, use find_knee out
    iscopy: bool
        whether to return a copy of adata or not, False by default
    """

    adata = adata.copy() if iscopy else adata
    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if evector_name not in adata.uns:
        raise ValueError(f"{evector_name} not in adata.uns")

    if eigen_n < 1:
        if "eigen_value_knee" in adata.uns.keys():
            eigen_n = adata.uns["eigen_value_knee"]
        else:
            eigen_n = knee_eigen(adata, eigens=re.sub(r"_vector$", r"_value", evector_name) , plot=False)
        print(f"eigen_n < 1, use knee_eigen to find the number of eigen vectors to use: {eigen_n}")
    if eigen_n == 1: ##TODO, if == 1 use first eigenvector is compatible or not
        print("eigen_n == 1 is too small, change to 2, use itsself as embedding")
        embedding = 'self'
    elif eigen_n == 2:
        print("eigen_n is 2, use itsself as embedding")
        embedding = 'self'

    if isinstance(M_flatten, str):
        M_flatten = adata.uns[M_flatten]

    mat_coor_flatten_trajectory = [adata.uns[evector_name][0:eigen_n, :] @ mat for mat in M_flatten.toarray()]

    adata.uns['trajs_harmonic_dm'] = np.vstack(mat_coor_flatten_trajectory)

    dm=None
    if embedding == "umap":
        dm = run_umap(mat_coor_flatten_trajectory)
    elif embedding == "pca":
        dm = run_pca(mat_coor_flatten_trajectory, n_components=2)
    elif embedding == "self":
        dm = np.vstack(mat_coor_flatten_trajectory)
    else:
        raise ValueError("embedding method not supported, only umap and pca are supported for now")
    adata.uns["trajs_dm"] = dm

    return adata if iscopy else None
#endf trajs_dm


def trajs_clustering(adata, embedding = 'trajs_harmonic_dm', clustering_method: str = "dbscan", iscopy=False, oname_basis='', **args,):
    """
    adata: AnnData
        AnnData object
    embedding: str
        the embedding for visualize the clustering results, "trajs_harmonic_dm" by default
    clustering_method: str
        the clustering method to use, options dbscan,leiden,louvain, "dbscan" by default
    iscopy: bool
        whether to return a copy of adata or not, False by default
    oname_basis: str
        the basis of the output name, "" by default
    args: dict
        the parameters for the clustering method
    """
    adata = adata.copy() if iscopy else adata
    dm = adata.uns[embedding]
    if clustering_method == "dbscan":
        clusters = dbscan(dm, **args)
    elif clustering_method == "leiden":
        clusters = leiden(dm, **args)
    elif clustering_method == "louvain":
        clusters = louvain(dm, **args)
    else:
        raise ValueError("clustering method not supported, only dbscan, leiden and louvain are supported for now")

    adata.uns[oname_basis + "trajs_clusters"] = clusters

    return adata if iscopy else None
#endf trajs_clustering



def harmonic_trajs_ranks(adata: AnnData,
                         group_name:str = 'group',
                         trajs_name = 'knn_trajs',
                         trajs_clusters = 'trajs_clusters',
                         trajs_use = 1000,
                         retain_clusters = [],
                         node_attribute = 'u',
                         top_n = 30,
                         min_kde_quant_rm = 0.1,
                         kde_sample_n = 1000,
                         verbose=True,
                         seed = 2022,
                         ):
    """
    calculate all harmonic trajectory groups end cluster(highest pseudotime score)
    check top n nodes belongs to which end cluster
    get a dict: {group: (end_cluster, topN_nodes)}
    return a dict including more than 1 trajectory groups end with the same cluster.
    {cluster: [g1,g2,g3,g4]}
    """
    import scipy.spatial
    from collections import Counter, defaultdict
    from scipy.stats import gaussian_kde

    np.random.seed(seed)
    trajs_use = min(trajs_use, len(adata.uns[trajs_name]))
    cluster_list = adata.uns[trajs_clusters]
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    assert(set(retain_clusters).issubset(set(cluster_list)))
    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)

    if verbose:
        print(f"check trajectory clusters: {retain_clusters}")

    d_node_freq = {}
    for cluster in retain_clusters:
        ### trajs are all nodes not edges
        itrajs = [i for i in np.where(cluster_list == cluster)[0]]
        if len(itrajs) > trajs_use:
            itrajs = np.random.choice(itrajs, trajs_use, replace=False)

        node_list = []
        for itraj in itrajs:
            node_list.extend(adata.uns[trajs_name][itraj])

        #kde = gaussian_kde(node_list)(node_list)
        #quantile = 0.1
        #kde_keep = np.where(kde > np.quantile(kde, quantile))[0]
        #node_list = np.array(node_list)[kde_keep]
        node_list = np.array(node_list)
        if node_attribute not in adata.obs.keys():
            assign_graph_node_attr_to_adata(adata, adata.uns["graph_basis"], from_attr=node_attribute, to_attr=node_attribute)

        u = adata.obs[node_attribute].iloc[list(Counter(node_list).keys())]
        topN = u.iloc[np.argsort(u)[::-1][:top_n]]
        d_node_freq[cluster] = topN

    ##record traject clusters with same ends
    d_c = defaultdict(list)# key(end_cluster), val(traject_clusters)
    d_c_detail = defaultdict(list)#
    for cluster in retain_clusters:
        c = Counter(adata.obs[group_name].loc[list(d_node_freq[cluster].index)])
        d_c[c.most_common()[0][0]].append(cluster)
        d_c_detail[cluster].append(c)
        if verbose:
            print("trajcluster: group&count", cluster, c.most_common()[0])

    adata.uns['end_cluster_counter_trajs'] = d_c_detail
    return {k:v for k,v in d_c.items() if len(v) > 1}
#endf

def undifferentiate_clusters(adata, trajs_clusters='trajs_clusters', harmonic_dm='trajs_harmonic_dm', std_threshold=2):
    from collections import Counter, defaultdict
    dm = adata.uns[harmonic_dm]
    cluster_set = np.unique(adata.uns[trajs_clusters])

    dic = defaultdict(float)
    for cluster in cluster_set:
        idx = np.where(adata.uns[trajs_clusters]  == cluster)[0]
        mtx = dm[idx, ]
        center = mtx.mean(axis=0, keepdims=True)
        d = scipy.spatial.distance.cdist(mtx, center)
        dic[cluster] = np.std(d)

    minn = min(dic.values())
    print(minn)
    dic = {k:(v/minn) for k,v in dic.items()}
    return {k for k,v in dic.items() if v > std_threshold}



def merge_trajectory_clusters(adata: AnnData,
                              group_name:str = 'group',
                              trajs_name = 'knn_trajs',
                              trajs_clusters = 'trajs_clusters',
                              trajs_use = 100,
                              retain_clusters = [],
                              node_attribute = 'u',
                              top_n = 30,
                              verbose=True,
                              dry_run=False,
                              iscopy = False,
                              ):

    adata = adata.copy() if iscopy else adata

    d_c = harmonic_trajs_ranks(adata, group_name, trajs_name, trajs_clusters, trajs_use, retain_clusters, node_attribute, top_n, verbose=verbose)

    for k,v in d_c.items():
        idx = np.where(np.isin(np.array(adata.uns[trajs_clusters]), v))
        if verbose:
            print(f"merge trajectory cluster {v} to be {v[0]}" )
        if not dry_run:
            adata.uns[trajs_clusters][idx] = v[0] ## assign to be the first one

    if not d_c  and verbose:
        print(f"no trajectory cluster to be merged")

    return adata if iscopy else None
#endf merge_trajectory_clusters

def trajactory_cluster_skewness(adata, trajs_name='knn_trajs', trajs_clusters = 'trajs_clusters', skewness=1, verbose=True):
    clusters = np.unique(adata.uns[trajs_clusters])
    s = set()
    for cluster in clusters:
        idx = np.where(adata.uns[trajs_clusters]  == cluster)[0]
        l = [len(adata.uns[trajs_name][i]) for i in idx]
        sk = scipy.stats.skew(l)
        if np.abs(sk) > skewness:
            if verbose:
                print(f"group {cluster}: skew abs({np.round(sk, 3)}) > {skewness}")
            s.add(cluster)
    return s

def unique_trajectory_clusters(adata: AnnData,
                              group_name:str = 'group',
                              trajs_name = 'knn_trajs',
                              trajs_clusters = 'trajs_clusters',
                              trajs_use = 100,
                              retain_clusters = [],
                              node_attribute = 'u',
                              top_n = 30,
                              verbose=True,
                              iscopy = False,
                              ):

    adata = adata.copy() if iscopy else adata

    d_c = harmonic_trajs_ranks(adata, group_name, trajs_name, trajs_clusters, trajs_use, retain_clusters, node_attribute, top_n, verbose=verbose)
    d_cluster_counts = Counter(adata.uns[trajs_clusters])
    rm_list = []
    for k,v in d_c.items():
        ## find the cluster which has largest number
        largest_idx = np.argmax(np.array([d_cluster_counts[i] for i in  v]))
        rm_clusters = list(np.delete(np.array(v), largest_idx))
        rm_list.extend(rm_clusters)
        if verbose:
            print(f"to remove clusters {rm_clusters} due to be duplicated" )
    if not d_c  and verbose:
        print(f"no duplicated trajectory cluster to be removed")
    remove_trajectory_clusters(adata, rm_list, trajs_clusters, trajs_name, iscopy=False, verbose=verbose)

    return adata if iscopy else None
#endf unique_trajectory_clusters



def select_trajectory_clusters(adata,
                               trajs_clusters="trajs_clusters",
                               trajs_name='knn_trajs',
                               rm_cluster_ratio=0.005,
                               manual_rm_clusters=[-1],
                               check_skewness = True,
                               skewness_threshold = 1,
                               iscopy=False,
                               verbose=True):
    """
    adata: AnnData
        AnnData object
    trajs_clusters: str
        the name of the clusters, "trajs_clusters" by default
    trajs_name: str
        the name of the trajs, "knn_trajs" by default
    rm_cluster_ratio: float
        smaller the ratio of the cluster number would be removed, 0.005(50 for 10,000) by default
    manual_rm_clusters: list
        the clusters to remove manually, [-1] by default
    iscopy: bool
        whether to return a copy of adata or not, False by default
    """
    from collections import Counter
    pass
    adata = adata.copy() if iscopy else adata

    ## 1. get remove number
    threshold = len(adata.uns[trajs_name]) * rm_cluster_ratio
    ## 2. get the cluster number to remove
    d_count = Counter(adata.uns[trajs_clusters])
    rm_clusters = [k for k, v in d_count.items() if v < threshold]
    ## 3. remove the clusters
    rm_clusters = rm_clusters + manual_rm_clusters

    if check_skewness:
        rm_clusters = rm_clusters + list(trajactory_cluster_skewness(adata, trajs_name, trajs_clusters, skewness_threshold, verbose=verbose))

    if verbose:
        print(f"clusters to remove({len(rm_clusters)})")
        print("\t".join([f"{str(i)}: {d_count.get(i, -1)}" for i in pd.unique(rm_clusters)]))
    remove_trajectory_clusters(adata, rm_clusters, trajs_clusters, trajs_name, iscopy=False, verbose=verbose)

    return adata if iscopy else None
#endf select_trajectory_clusters

def remove_trajectory_clusters(adata,
                               rm_clusters,
                               trajs_clusters="trajs_clusters",
                               trajs_name='knn_trajs',
                               iscopy=False,
                               verbose=True):

    adata = adata.copy() if iscopy else adata

    rm_idxs = np.where(np.isin(np.array(adata.uns[trajs_clusters]), rm_clusters))[0]

    if verbose:
        print(f"remove clusters: #removed_trajectories({len(rm_idxs)}), #remain_trajectories({len(adata.uns[trajs_name]) - len(rm_idxs)})")

    ## 5. update all including:
        ## a. clusters
        ## b. trajs
        ## c. full_traj_matrix
        ## d. full_traj_matrix_flatten
        ## e. full_traj_matrix_flatten_norm if exists
        ## f. trajs_harmonic_dm
        ## g. trajs_dm
    if verbose:
        print("updateing..")
        print("clusters", trajs_clusters)
        print("trajs", trajs_name)
        print("full_traj_matrix")
        print("full_traj_matrix_flatten")
        print("full_traj_matrix_flatten_norm")
        print("trajs_harmonic_dm")
        print("trajs_dm")

    adata.uns[trajs_clusters] = np.delete(np.array(adata.uns[trajs_clusters], dtype=object), rm_idxs)
    adata.uns[trajs_name] = np.delete(np.array(adata.uns[trajs_name], dtype=object), rm_idxs)

    keep_idx = np.delete(np.arange(len(adata.uns['full_traj_matrix'])), rm_idxs)
    adata.uns['full_traj_matrix'] = np.delete(np.array(adata.uns['full_traj_matrix']), rm_idxs)
    adata.uns['full_traj_matrix_flatten'] = adata.uns['full_traj_matrix_flatten'][keep_idx, :]
    if 'full_traj_matrix_flatten_norm' in adata.uns.keys():
        adata.uns['full_traj_matrix_flatten_norm'] = adata.uns['full_traj_matrix_flatten_norm'][keep_idx, :]
    adata.uns['trajs_harmonic_dm'] = np.delete(np.array(adata.uns['trajs_harmonic_dm']), rm_idxs, axis=0)
    adata.uns['trajs_dm'] = np.delete(np.array(adata.uns['trajs_dm']), rm_idxs, axis=0)

    return adata if iscopy else None
#endf remove_trajectory_clusters


def G_random_climb(g:nx.Graph, attr:str='u', roots_ratio:float=0.1, n:int=10000, seeds:int=2022) -> list:
    """
    random climb of a graph according to the attr

    Parameters
    --------
    g: Graph
    attr: node attribute for climbbing
    roots_ratio: choose the ratio of all nodes as starts
    n: how many trajectories to generate
    """
    random.seed(seeds)
    dattr = nx.get_node_attributes(g, attr)
    nodes = [k for k, v in sorted(dattr.items(), key=lambda item: item[1])]
    topn = int(roots_ratio*len(nodes))

    trajs = []
    for i in range(n):
        fromi = random.randrange(0, topn)
        from_vertex = nodes[fromi]
        a_traj = [from_vertex, ]
        while True:
            tovertices = list(g.neighbors(from_vertex))
            u_from = dattr[from_vertex]
            d_u_to = {to:dattr[to] for to in tovertices if dattr[to] > u_from}

            to_select=-1
            if len(d_u_to) == 0:
                break
            elif len(d_u_to) == 1:
                to_select = list(d_u_to.keys())[0]
            else:
                to_select = random.choices(list(d_u_to.keys()), weights=list(d_u_to.values()), k=1)[0]
            from_vertex = to_select
            if to_select != -1:
                a_traj.append(to_select)
        trajs.append(a_traj)
    return trajs



def G_shortest_path_edge(g, an_edge):
    if g.has_edge(an_edge[0], an_edge[1]):
        return [an_edge[0],]
    return nx.shortest_path(g, source=an_edge[0], target=an_edge[1])[:-1]
#endf

def G_random_climb_knn(g:nx.Graph, knn_edges, attr:str='u', roots_ratio:float=0.1, n:int=10000, seeds:int=2022) -> list:
    """
    1st: Random climb using KNN graph to construct a trajectory.
    2nd: Check each edge of the trajectory, if the edge does not belong to the edge of the graph G, find shotest path
    3rd: Get the final trajectory

    In step 2nd, the edges consist of the nodes,
    that is we care about only if edge exists, and store the direction by the nodes order

    Parameters
    ------------
    g: Graph
    knn_edges: knn edges
    attr: node attribute for climbbing
    roots_ratio: choose the ratio of all nodes as starts
    n: how many trajectories to generate
    """
    knn_g = nx.create_empty_copy(g)
    knn_g.add_edges_from(knn_edges)
    knn_trajs = G_random_climb(knn_g, attr=attr, roots_ratio=roots_ratio, n=n, seeds=seeds)
    for i in trange(len(knn_trajs)):
        the_last = knn_trajs[i][-1]
        lol = [G_shortest_path_edge(g, x) for x in pairwise(knn_trajs[i])]
        knn_trajs[i] = [y for x in lol for y in x] + [the_last] ## flatten

    return knn_trajs
#endf

def L_trajectory_class(traj:list, groups, last_n=10, min_prop=0.8, all_n=3):
    """
    Parameters
    ---------
    traj: a trajectory
    last_n: check last n nodes of a trajectory
    min_prop: check if the class satisfy the proportions
    all_n: all these last n nodes must be in the majority class
    """
    maj   = [groups[x] for x in traj[-last_n:-1]]
    end_e = [groups[x] for x in traj[-all_n:-1]]
    maj_c = L_majority_proportion(maj, min_prop)
    if maj_c and all(maj_c == x for x in end_e):
        return maj_c
    return None


def L_majority_proportion(lst, min_prop=0.8):
    d = Counter(lst)
    total = sum(d.values())
    for key in d.keys():
        p = d[key]*1.0/total
        if p > min_prop:
            return key
    return None
#endf majority_proportion

def L_distribute_traj(trajs, groups):
    d_trajs = defaultdict(list)
    for i in range(0, len(trajs)):
        c = L_trajectory_class(trajs[i], groups)
        if not c: continue
        d_trajs[c].append(trajs[i])
    return d_trajs
#endf distribute_traj


def knee_points(mat_coord_Hspace, trajs):
    assert(len(mat_coord_Hspace) == len(trajs))
    d_branch = defaultdict(int)
    cumsums = list(map(lambda i: [np.cumsum(i[0]), np.cumsum(i[1])], mat_coord_Hspace))
    for i in range(len(trajs)):
        idx = find_knee(cumsums[i][0], cumsums[i][1])
        d_branch[trajs[i][idx]] += 1
        d_branch[trajs[i][idx+1]] += 1
    return d_branch


def G_full_trajectory_matrix(graph: nx.Graph, mat_traj, elist, elist_dict, edge_w=None) -> List[csr_matrix]:
    """
    import from https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/
    """
    mat_vec_e = []
    for count, j in enumerate(mat_traj):
        if len(j) == 0:
            #print(f"{count}: No Trajectory")
            continue

        data = []
        row_ind = []
        col_ind = []

        for i, (x, y) in enumerate(j):
            assert (x, y) in elist_dict or (y, x) in elist_dict  # TODO
            assert graph.has_edge(x, y)

            if x < y:
                idx = elist_dict[(x, y)]
                row_ind.append(idx)
                col_ind.append(i)
                if edge_w is not None:
                    data.append(edge_w[idx])
                else:
                    data.append(1)
            if y < x:
                idx = elist_dict[(y, x)]
                row_ind.append(idx)
                col_ind.append(i)
                if edge_w is not None:
                    data.append(-1 * edge_w[idx])
                else:
                    data.append(-1)

        mat_temp = csr_matrix((data, (row_ind, col_ind)), shape=(
            elist.shape[0], len(j)), dtype=np.float32)# int8
        mat_vec_e.append(mat_temp)
    return mat_vec_e

def L_flatten_trajectory_matrix(M_full) -> np.ndarray:
    """
    import from https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/
    """
    flattened = map(lambda mat_tmp: mat_tmp.sum(axis=1), M_full)
    return scipy.sparse.csr_matrix(np.array(list(flattened)).squeeze())

def L_flatten_trajectory_matrix_norm(M_full) -> np.ndarray:
    flattened = map(lambda mat_tmp: mat_tmp.sum(axis=1)/mat_tmp.shape[1], M_full)
    return scipy.sparse.csr_matrix(np.array(list(flattened)).squeeze())



def M_create_matrix_coordinates_trajectory_Hspace(H, M_full):
    """
    import from https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/
    """
    return [H @ mat for mat in M_full]



def M_create_matrix_coordinates_trajectory_Hspace_dm(mat_coord_Hspace, reductioin='pca'):
    #mat_coord_Hspace = phlower.tl.M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:20], adata.uns[full_traj_matrix])
    #phlower.pl.M_plot_trajectory_harmonic_lines(dm_mat_list, adata.uns['annotation'], sample_ratio=0.1)

    trajs_pos = [i.shape[1] for i in mat_coord_Hspace]

    h_mat = np.hstack(mat_coord_Hspace).T
    if reductioin == 'pca':
        dm_mat = run_pca(h_mat)
    elif reductioin == 'umap':
        dm_mat = run_umap(h_mat)

    last = 0
    dm_mat_list = []
    for i in np.cumsum(trajs_pos):
        dm_mat_list.append(dm_mat[last:i, ])
        last = i

    return dm_mat_list



def curated_paths(adata=None,
                  graph_name='X_pca_ddhodge_g_triangulation',
                  obs_ct = 'group',
                  obs_time = 'u',
                  pca='harmony_X',
                  n_trajs=10000,
                  start_n = 100,
                  end_n = 300,
                  middle=5,
                  root = "HSPC_MPP",
                  ends = ['cDC_2', "Monocytes", "B_cells", "Erythroid_cells", "Macrophages", 'Neutrophils', 'Proliferating_granulocytes'],
                  edge_weight='weight',
                  n_neighbors = 30,
                  random_seed = 1,
                  ):
    """
    specify start and ends to create trajectories

    1. select middle number of middle point form the shortest path
    2. randomly select a neighbor from the middle point to form new paths

    Parameters
    ----------

    Return
    ------
    """

    random.seed(random_seed)

    ## shuffle the top start_n cells
    start_cells = top_n_cells_celltype(adata, slot=obs_ct, val=root, time=obs_time, n=start_n, top="smallest")
    sstart_cells = np.random.choice(start_cells, start_n, replace=True)

    end_cells_all = []
    for end in ends:
        #print(end)
        end_cells = list(top_n_cells_celltype(adata, slot=obs_ct, val=end, time=obs_time, n=end_n, top="largest"))
        end_cells_all.extend(end_cells)

    ##shuffle ends
    send_cells_all = np.random.choice(end_cells_all, len(end_cells_all), replace=True)

    ## create nearest neighbors matrix using PCA DR
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(np.array(adata.obsm[pca]))
    indist, inind = nbrs.kneighbors(np.array(adata.obsm[pca]))

    ## get all pairs starts and ends
    pairs = itertools.product(sstart_cells, send_cells_all)
    trajs = []
    for i,j in tqdm.tqdm(pairs, total=n_trajs):
        #print(i)
        start = np.where(adata.obs_names == i)[0][0]
        end = np.where(adata.obs_names == j)[0][0]
        atraj = nx.shortest_path(adata.uns[graph_name], start, end, weight=edge_weight)
        length=len(atraj)
        pmiddles = [int(i) for i in np.linspace(0, length, middle+2)][1:-1]
        pass_node = [start]
        for pmid in pmiddles:
            imid = atraj[pmid]
            mids = inind[imid, :][1:]
            rbr = np.random.choice(mids, 1, replace=True)[0]
            pass_node.append(rbr) ## add middle random nodes
        pass_node.append(end)

        atrajn = shortest_trajectory(adata.uns[graph_name], pass_node)
        trajs.append(atrajn)
        if len(trajs) >= n_trajs:
            break
    return trajs


def top_n_cells_celltype(adata, slot='group', val='HSPC_MPP',time='u', n=40, top='smallest'):
    """
    return top largest or smallest cell names of a celltype according the time
    """
    #print(slot, val)
    idx = adata.obs[slot] == val
    #print(idx)
    #
    if top=='smallest':
        theta = adata.obs[time][idx][np.argsort(adata.obs[time][idx])[min(n, len(adata.obs[time][idx]))]] # min_u
        sub_idx = np.where(adata.obs[time][idx] <= theta)[0]
    else:
        theta = adata.obs[time][idx][np.argsort(adata.obs[time][idx])[-min(n, len(adata.obs[time][idx]))]] # max_u
        sub_idx = np.where(adata.obs[time][idx] >= theta)[0]

    #print(np.where(adata.obs[time][idx] < theta)[0])
    cells = adata.obs_names[idx][sub_idx]
    return cells
#endf top_n_cells_celltype

def shortest_trajectory(graph, pass_nodes):
    """
    use a series of pass_nodes to find shortest paths between each pair of nodes

    Return
    ------
    lst: list of all points in the paths
    """
    lst = []
    for i, pair in enumerate(pairwise(pass_nodes)):
        x = list(nx.shortest_path(graph, pair[0], pair[1], weight='weight'))
        if i == len(pass_nodes)-2: ## last one
            lst.extend(x)
        else:
            lst.extend(x[:-1])
    return lst


def detect_short_trajectory_groups(adata, trajectories='knn_trajs', cluster_name="h_trajs_clusters", min_len=10, verbose=False):
    """
    after clustering, if there are some trajectory groups are too short, list them
    """

    all_clusters = set(adata.uns[cluster_name])
    ret_list = []
    if verbose:
        print("List trajectories clusters with median length <= 10")
    for cluster in all_clusters:
        cluster_i = np.where(np.array(adata.uns[cluster_name]) == cluster)[0]
        median_len = np.median([len(adata.uns[trajectories][i]) for i in cluster_i])
        if verbose:
            print(f"Cluster {cluster} length: {median_len}")
        if median_len <= min_len:
            ret_list.append(cluster)
    return ret_list


def cumsum_Hspace(mat_coord_Hspace, dims:List):
    """
    cumsum with spaces in dims
    """
    return [np.array([np.cumsum(m[j, :]) for j in dims]).T for m in  mat_coord_Hspace]
#cumsums
