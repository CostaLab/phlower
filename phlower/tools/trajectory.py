##TODO
## remove noise clusters labelled by -1


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
from ..util import pairwise, find_knee, tuple_increase, is_node_attr_existing


def random_climb_knn(adata,
                     graph_name = "X_dm_ddhodge_g_triangulation",
                     A = "X_dm_ddhodge_A",
                     W = "X_dm_ddhodge_W",
                     knn_edges_k = 9,
                     attr:str='u',
                     roots_ratio:float=0.1,
                     n:int=10000,
                     iscopy=False,
                     traj_name = None,
                     seeds:int=2022):

    adata = adata.copy() if iscopy else adata

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
    knn_edges = adjedges(adata.uns[A], adata.uns[W], knn_edges_k)
    knn_edges = [tuple_increase(i,j) for (i,j) in knn_edges]
    knn_trajs = G_random_climb_knn(g, knn_edges, attr=attr, roots_ratio=roots_ratio, n=n, seeds=seeds)
    if traj_name is None:
        traj_name = f"knn_trajs"

    adata.uns[traj_name] = knn_trajs

    return adata if iscopy else None


def full_trajectory_matrix(adata: AnnData,
                           graph_name: str = "X_dm_ddhodge_g_triangulation_circle",
                           trajs : Union[str, List[List[int]]] = "knn_trajs",
                           edge_w : List = None,
                           oname_basis : str = "",
                           iscopy = False,
                           ):
    adata = adata.copy() if iscopy else adata

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")


    if isinstance(trajs, str):
        if trajs not in adata.uns:
            raise ValueError(f"{trajs} not in adata.uns")
        trajs = adata.uns[trajs]

    g = adata.uns[graph_name]
    elist = np.array([(x[0], x[1]) for x in g.edges()])
    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}
    M_full = G_full_trajectory_matrix(g, map(lambda path: list(edges_on_path(path)), chain.from_iterable([trajs])), elist, elist_dict)
    adata.uns[oname_basis + "full_traj_matrix"] = M_full
    adata.uns[oname_basis + "full_traj_matrix_flatten"] = L_flatten_trajectory_matrix(M_full)


    return adata if iscopy else None


def trajs_dm(adata,
             evector_name: str = "X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
             M_flatten: Union[str, np.ndarray] = "full_traj_matrix_flatten",
             embedding = 'umap',
             eig_num: int = 2,
             iscopy=False,
             **args,
             ):

    adata = adata.copy() if iscopy else adata

    if evector_name not in adata.uns:
        raise ValueError(f"{evector_name} not in adata.uns")

    if eig_num < 2:
        raise ValueError(f"eig_num is {eig_num}, should be >= 2")

    if isinstance(M_flatten, str):
        M_flatten = adata.uns[M_flatten]

    mat_coor_flatten_trajectory = [adata.uns[evector_name][0:eig_num, :] @ mat for mat in M_flatten]

    adata.uns['trajs_harmonic_dm'] = np.vstack(mat_coor_flatten_trajectory)

    dm=None
    if embedding == "umap":
        dm = run_umap(mat_coor_flatten_trajectory)
    elif embedding == "pca":
        dm = run_pca(mat_coor_flatten_trajectory, n_components=2)
    else:
        raise ValueError("embedding method not supported, only umap and pca are supported for now")
    adata.uns["trajs_dm"] = dm

    return adata if iscopy else None
#endf trajs_dm


def trajs_clustering(adata, embedding = 'trajs_dm', clustering_method: str = "dbscan", iscopy=False, oname_basis='', **args,):
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
            print(f"{count}: No Trajectory")
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

def L_flatten_trajectory_matrix(H_full) -> np.ndarray:
    """
    import from https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/
    """
    flattened = map(lambda mat_tmp: mat_tmp.sum(axis=1), H_full)
    return np.array(list(flattened)).squeeze()


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

