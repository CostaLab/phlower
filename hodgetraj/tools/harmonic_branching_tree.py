import math
import pandas as pd
import numpy as np
import networkx as nx
from anndata import AnnData
from sklearn.neighbors import KDTree
from tqdm import trange
from collections import Counter, defaultdict
from ..util import bsplit
from .trajectory import M_create_matrix_coordinates_trajectory_Hspace

from ..external.stream_extra import (add_pos_to_graph,
                                     dfs_from_leaf,
                                     extract_branches,
                                     construct_stream_tree,
                                     add_branch_info,
                                     project_cells_to_g_fate_tree,
                                     calculate_pseudotime,
                                     )




def atraj_edges_split(cumsum, itraj, row_clusters, row_edges, indices, dic_traj_starts_idx, dic_edges_counts,isprint=False):

    dic_keys_counts = defaultdict(int)
    if not isinstance(row_clusters, np.ndarray):
        row_clusters = np.array(row_clusters)
    for i in range(cumsum[0].shape[0]):
        row_idx = i+dic_traj_starts_idx[itraj]
        nn = (row_clusters[indices[row_idx]])
        dd = Counter(nn)
        for k,v in list(dd.items()):
            if v <3: ## only 2 present in NNs, remove it
                del dd[k]

        t_k = tuple(set(dd.keys()))
        if t_k not in dic_edges_counts:
            dic_edges_counts[t_k] = defaultdict(int)
        dic_edges_counts[t_k][row_edges[row_idx]] +=1

        dic_keys_counts[t_k] += 1

        if isprint:
            print(i, len(dd.keys()) ,dd)
    return dic_keys_counts



def create_hstream_tree(adata: AnnData,
                       fate_tree: str = 'fate_tree',
                       layout_name: str = 'X_dm_ddhodge_g',
                       iscopy: bool = False,
                       ):

    adata = adata.copy() if iscopy else adata
    g = adata.uns[fate_tree]
    g_pos   = adata.uns[fate_tree].to_undirected()
    dic_br  =  extract_branches(g_pos.to_undirected())
    dic_br  =  add_branch_info(g_pos.to_undirected(), dic_br)
    g_br    =  construct_stream_tree(dic_br, g_pos)

    adata.uns['g_fate_tree'] = g_pos
    adata.uns['stream_tree'] = g_br
    layouts = adata.obsm[layout_name]
    if type(layouts) == dict:
        layouts = np.array([layouts[x] for x in range(max(layouts.keys()) + 1)])
    project_cells_to_g_fate_tree(adata, layout_name=layout_name)
    calculate_pseudotime(adata)

    return adata if iscopy else None
#endf create_stream_tree


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

def htraj_fair_matrix(adata=None,
                 evector_name="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                 full_traj_matrix = "full_traj_matrix",
                 trajs_clusters = 'trajs_clusters',
                 trajs_each = 100,
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
    from IPython.core.debugger import set_trace
    #set_trace()

    m_full_traj_matrix = adata.uns[full_traj_matrix]
    trajs_each = min(trajs_each, len(m_full_traj_matrix))
    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:eigen_n, :], adata.uns[full_traj_matrix])
    cumsums = list(map(lambda i: [np.cumsum(j) for j in i ], mat_coord_Hspace))

    cluster_counter = Counter(adata.uns[trajs_clusters])
    trajs_each = min(trajs_each, min(cluster_counter.values()))
    idxs = []
    for key in cluster_counter:
        idxs.extend(np.where(np.array(adata.uns[trajs_clusters]) == key)[0][0:trajs_each])
    cumsums = [cumsums[i] for i in idxs]

    mat = None
    row_ids = np.array([], dtype=int)
    row_clusters = []
    row_edges = [] ## store trajectory edges info
    dic_traj_starts_idx = {0:0} #store where to start for each trajectory

    trajs_mtxs = np.array(adata.uns[full_traj_matrix])[idxs]
    for itraj in trange(len(cumsums)):
        nmtx = np.vstack(cumsums[itraj]).T
        traj_mtx = trajs_mtxs[itraj]
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
                  graph_name: str = 'X_dm_ddhodge_g_triangulation_circle',
                  layout_name: str = 'X_dm_ddhodge_g',
                  evector_name="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                  full_traj_matrix = "full_traj_matrix",
                  trajs_clusters = 'annotation',
                  node_attribute = 'u',
                  k = 100,
                  eigen_n = 20,
                  bucket_number=3,
                  trajs_use = 10000,
                  tree_name = 'htree',
                  **args
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
    distances, indices = traj_knn(mat, k=k, **args)

    print("counting the trajectory groups...")
    keys_counters = defaultdict(int)
    dic_edges_counts = {} ## store the edge counts for each group
    for i in trange(0, min(trajs_use, len(cumsums))):
        dic = atraj_edges_split(cumsums[i], i, row_clusters, row_edges,indices, dic_traj_starts_idx, dic_edges_counts, isprint=False)
        for key in dic:
            keys_counters[key] += dic[key]

    full_list = list(set(adata.uns[trajs_clusters]))


    dic_u = edge_mid_attribute(adata, graph_name, node_attribute)
    dff = edges_distribute_dic(dic_edges_counts[tuple(full_list)], dic_u, 2)

    tree_list=[]
    htree = nx.DiGraph()
    last_node = tree_add_nodes(htree, dff, bucket_idx_name = "bucket_idx", name_idx=0, ttype=tuple(full_list))

    create_detail_harmonic_tree_list(full_list, tree_list, htree, keys_counters, dic_edges_counts, dic_u, last_node, bucket_number=bucket_number)
    #for node in htree.nodes():
    #    htree.nodes[node]['edges'] = list(dic_edges_counts[node].items())

    dic_avg_edge_pos = edge_mid_points(adata, graph_name, layout_name)
    htree = add_pos_to_graph_edge(htree, dic_avg_edge_pos)

    return htree


def edges_distribute_dic(dic_edges, dic_u, min_u=0, bucket_number=5, bucket_idx_name = "bucket_idx"):
    #print(dic_edges)
    dff = pd.DataFrame(dic_edges, index=('counts',)).T
    #print(dff)
    dff['edges'] = dff.index
    dff['u'] = [ dic_u[i] for i in dff.index]
    dff = dff[dff['u']>=min_u]
    dff['rank'] = dff['u'].rank()
    bins = np.linspace(dff['rank'].min(), dff['rank'].max(), bucket_number+1)
    if len(set(bins))== 1:
        #bins = np.array([dff['rank'].min(), dff['rank'].max()])
        dff['buckets'] = dff['rank']
    else:
        dff['buckets'] = pd.cut(dff['rank'], bins , include_lowest=True)
    d_buckets = {v:i for  i, v in enumerate(sorted(set(dff.buckets)))}
    dff[bucket_idx_name] = dff['buckets'].apply(lambda x: d_buckets[x])

    return dff



def edges_distribute_buckets_num(dic_edges_counts, key_list, dic_u, min_bucket_number=2,):
    """
    buckets number is determined by the u range sqrt(divided by min_bucket_number)
    """
    max_min_range = 1000000 ##big enough
    for key in key_list:
        key = (key, )
        dff = pd.DataFrame(dic_edges_counts[key], index=('counts',)).T
        dff['edges'] = dff.index
        dff['u'] = [ dic_u[i] for i in dff.index]
        u_range = dff['u'].max() - dff['u'].min()
        max_min_range = min(max_min_range, u_range)
        print(key, u_range)

    dic_buckets = {}
    for key in key_list:
        key = (key, )
        dff = pd.DataFrame(dic_edges_counts[key], index=('counts',)).T
        dff['edges'] = dff.index
        dff['u'] = [ dic_u[i] for i in dff.index]
        dic_buckets[key] = int(round(max(min_bucket_number, min_bucket_number*math.sqrt((dff['u'].max() - dff['u'].min()) / max_min_range))))

    return dic_buckets



def edges_distribute_auto(dic_edges_counts, key, dic_u, bucket_number=5, bucket_idx_name = "bucket_idx"):
    print("key", key)
    dff = pd.DataFrame(dic_edges_counts[key], index=('counts',)).T
    dff['edges'] = dff.index
    dff['u'] = [ dic_u[i] for i in dff.index]
    dff['rank'] = dff['u'].rank()
    bins = np.linspace(dff['rank'].min(), dff['rank'].max(), bucket_number+1)
    dff['buckets'] = pd.cut(dff['rank'], bins , include_lowest=True)
    d_buckets = {v:i for  i, v in enumerate(sorted(set(dff.buckets)))}
    dff[bucket_idx_name] = dff['buckets'].apply(lambda x: d_buckets[x])

    return dff






def tree_add_nodes(htree, dff, bucket_idx_name = "bucket_idx", name_idx=0, ttype=None, parent=None):
    if parent:
        htree.add_edge(parent, f"{name_idx}_{0}")

    maxx = max(dff[bucket_idx_name])
    #print(maxx)
    for i in range(maxx):
        nodes_1 = f"{name_idx}_{i}"
        nodes_2 = f"{name_idx}_{i+1}"
        htree.add_edge(nodes_1, nodes_2)
        htree.nodes[nodes_1]['edges'] = [(row['edges'], row['counts']) for idx, row in dff.loc[dff[bucket_idx_name] == i, ("edges", 'counts')].iterrows()]
        htree.nodes[nodes_2]['edges'] = [(row['edges'], row['counts']) for idx, row in dff.loc[dff[bucket_idx_name] == i+1, ("edges", 'counts')].iterrows()]
        htree.nodes[nodes_1]['ttype'] = ttype
        htree.nodes[nodes_2]['ttype'] = ttype

    return nodes_2


def edge_mid_points(adata: AnnData,
                    graph_name: str = 'X_dm_ddhodge_g_triangulation_circle',
                    layout_name: str = 'X_dm_ddhodge_g',
                    ):
    """
    return middle points of all edges
    """
    elist = np.array([(x[0], x[1]) for x in adata.uns[graph_name].edges()])
    dic = {}
    for i in range(elist.shape[0]):
        dic[i] =np.average( (adata.obsm[layout_name][elist[i][0]], adata.obsm[layout_name][elist[i][1]]), axis=0)
    return dic

def edge_mid_attribute(adata: AnnData,
                       graph_name: str = 'X_dm_ddhodge_g_triangulation_circle',
                       node_attribute = 'u',
                       ):
    """
    return middle time of all edges
    """
    #assert() attribute type
    elist = np.array([(x[0], x[1]) for x in adata.uns[graph_name].edges()])
    dic = {}
    u_score = nx.get_node_attributes(adata.uns[graph_name], node_attribute)
    for i in range(elist.shape[0]):
        dic[i] = (u_score[elist[i][0]] + u_score[elist[i][1]])/2
    return dic





## add position using splited time of attribute a.
## add the middle points between edges
## can use the pseudo time information to add the middle points
##



def add_pos_to_graph_edge(graph, layouts, weight_power=2,iscopy=False):
    """
    using an edge information to add the position of the htree
    """

    assert(type(layouts) == type({}) or type(layouts)==type([]) or type(layouts)==type(np.array([])))
    dict_pos = dict()
    if iscopy:
        graph_out = copy.deepcopy(graph)
    else:
        graph_out = graph

    for node in graph_out.nodes():
        if 'edges' not in graph_out.nodes[node]:
            raise ValueError(f"no edges in the node {node}")
        edge_data = graph_out.nodes[node]['edges']
        if isinstance(edge_data, defaultdict) or isinstance(edge_data, dict):
            edge_data = tuple(edge_data.items())
        #print('node', node)
        #print('e data', edge_data)
        edge = [i[0] for i in edge_data]
        weight = [np.power(i[1], weight_power) for i in edge_data]
        dict_pos[node] = np.average([layouts[x] for x in edge], axis=0, weights=weight)
        #dict_pos[node] = np.average([layouts[x] for x in edge], axis=0)

    #print(dict_pos)
    nx.set_node_attributes(graph_out,values=dict_pos,name='pos')
    return graph_out
#endf add_pos_to_graph_eddge


def trajs_travel_stat(cumsums, row_clusters, row_edges, indices, dic_traj_starts_idx, dic_edges_counts,isprint=False):
    change_dict = defaultdict(int)
    edges_branching_dict = {}
    dic_keys_counts = defaultdict(int)


    for itraj in trange(len(cumsums)):
        cumsum = cumsums[itraj]
        if not isinstance(row_clusters, np.ndarray):
            row_clusters = np.array(row_clusters)

        dd_list = []
        edge_list = []
        for i in range(cumsum[0].shape[0]):
            row_idx = i+dic_traj_starts_idx[itraj]
            nn = (row_clusters[indices[row_idx]])
            #print(nn)
            dd = Counter(nn)
            for k,v in list(dd.items()):
                if v <4: ## only 2 present in NNs, remove it
                    del dd[k]
            if i > 0 and len(dd.keys()) > len(dd_list[-1].keys()): ## remove increase number of trajectory cell types.
                continue

            dd_list.append(dd)
            #print(dd)
            edge_list.append(row_edges[row_idx])
            t_k = tuple(sorted(list(dd.keys())))
            if t_k not in dic_edges_counts:
                dic_edges_counts[t_k] = defaultdict(int)
            dic_edges_counts[t_k][row_edges[row_idx]] +=1
            dic_keys_counts[t_k] += 1


        ## count branching
        for i in range(1, len(dd_list)):
            if len(dd_list[i].keys()) < len(dd_list[i-1].keys()):
                change_dict[(tuple(sorted(list(dd_list[i-1].keys()))), tuple(sorted(list(dd_list[i].keys()))))] += 1
                if (tuple(sorted(list(dd_list[i-1].keys()))), tuple(sorted(list(dd_list[i].keys())))) not in edges_branching_dict:
                    edges_branching_dict[(tuple(sorted(list(dd_list[i-1].keys()))), tuple(sorted(list(dd_list[i].keys()))))] = defaultdict(int)
                edges_branching_dict[(tuple(sorted(list(dd_list[i-1].keys()))), tuple(sorted(list(dd_list[i].keys()))))][edge_list[i]] +=1

    return change_dict, edges_branching_dict, dic_keys_counts



def vertices_split(adata: AnnData,

                   ):
    """
    split each vertex into several vertices based the pseudotime of the edges
    Each vertices now is a meta vertex, we split these nodes by the times
    """



    pass




### should create using the end points, so that we can add middle points from the starts.
def create_detail_harmonic_tree_list(lst, tree_list, htree, keys_counters, dic_edges_counts, dic_u, last_node,  times_diff=50, name_idx=0, bucket_number=2):
    #print(max(dic_u.keys()), "--------------------------------------")
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

        name_idx +=1

        dff = edges_distribute_dic(dic_edges_counts[tuple(lst1)], dic_u, bucket_number=bucket_number, bucket_idx_name='bucket_idx')
        last_node1 = tree_add_nodes(htree, dff, "bucket_idx", name_idx+1, parent=last_node)
        #print("add 1", htree.edges())

        dff = edges_distribute_dic(dic_edges_counts[tuple(lst2)], dic_u, bucket_number=bucket_number, bucket_idx_name='bucket_idx')
        last_node2 = tree_add_nodes(htree, dff, "bucket_idx", name_idx+2, parent=last_node)
        #print("add 2", htree.edges())
        #htree.add_edge(tuple(lst), lst1)
        #htree.add_edge(tuple(lst), lst2)


        tree_list.append((lst1, lst2))
        create_detail_harmonic_tree_list(lst1, tree_list, htree, keys_counters, dic_edges_counts, dic_u, last_node1, times_diff, name_idx=name_idx+1, bucket_number=bucket_number)
        create_detail_harmonic_tree_list(lst2, tree_list, htree, keys_counters, dic_edges_counts, dic_u, last_node2, times_diff, name_idx=name_idx+2, bucket_number=bucket_number)
    else:
        return



### should create using the end points, so that we can add middle points from the starts.
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
        create_harmonic_tree_list(lst1, tree_list, htree, keys_counters, times_diff)
        create_harmonic_tree_list(lst2, tree_list, htree, keys_counters, times_diff)
    else:
        return



