import math
import copy
import sklearn
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from anndata import AnnData
from scipy.spatial import distance
from scipy.stats import gaussian_kde
from sklearn.neighbors import KDTree
from tqdm import trange
from networkx.algorithms.components.connected import connected_components
from collections import Counter, defaultdict
from ..util import bsplit, pairwise, term_frequency_cosine, find_cut_point, find_cut_point_bu

from .trajectory import M_create_matrix_coordinates_trajectory_Hspace

from ..external.stream_extra import (add_pos_to_graph,
                                     dfs_from_leaf,
                                     extract_branches,
                                     construct_stream_tree,
                                     add_branch_info,
                                     project_cells_to_g_fate_tree,
                                     calculate_pseudotime,
                                     )





def create_bstream_tree(adata: AnnData,
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

def get_root_bins(ddf, node_name, start, end):
    """
    get the root bins
    """
    all_nodes = set(node_name)
    root_bins = []

    for i in range(start, end):
        for node in all_nodes:
            df = ddf[node]

            if i in df['ubin'].values:
                root_bins.append(i)
                break

    return sorted(root_bins)

def create_detail_tree(htree, root, ddf):
    travel_edges = list(nx.bfs_tree(htree, root).edges())
    fate_tree = nx.DiGraph()
    root = None
    for i, edges in enumerate(travel_edges):
        n0 = edges[0]
        n1 = edges[1]
        t1 = htree.nodes[n0].get('time', -1)
        t2 = htree.nodes[n1].get('time', -1)
        if i == 0: #root
            curr_tm = 0
            # look up all merged trajectoris, see if the time point exists
            rest_ubins = get_root_bins(ddf, n0, 0, t1+1)
            for tm in rest_ubins[1:]:
                fate_tree.add_edge(((n0, curr_tm)), ((n0, tm)))
                curr_tm = tm
            root = (n0, 0)

        if t2 == -1: ## to the ends
            rest_ubins = sorted([i for i in set(ddf[n1[0]]['ubin']) if i > t1])
            #print(rest_ubins)
            curr_tm = rest_ubins[0]
            fate_tree.add_edge((n0, t1), (n1, curr_tm))
            for tm in rest_ubins[1:]:
                fate_tree.add_edge(((n1, curr_tm)), ((n1, tm)))
                curr_tm = tm
        else: ## middle branches
            #for tm in range(t1, t2):
            ubin_sets = [set(ddf[x]['ubin']) for x in n1]
            rest_ubins = [i for i in set.union(*ubin_sets) if  t1 < i <= t2]
            #print(rest_ubins)
            curr_tm = rest_ubins[0]
            fate_tree.add_edge(((n0, t1)), ((n1, curr_tm)))
            for tm in rest_ubins[1:]:
                fate_tree.add_edge(((n1, curr_tm)), ((n1, tm)))
                curr_tm = tm

    fate_tree = add_node_info(fate_tree, ddf, root)
    fate_tree = relabel_tree(fate_tree, root)

    return fate_tree



def relabel_tree(fate_tree, root):
    """
    relabel a tree to be astring due to its complicated name giving rise to error of function pairwise_distances_argmin_min
    """
    travel_nodes = list(nx.bfs_tree(fate_tree, root).nodes())
    all_pre = sorted(list({i[0] for i in  travel_nodes}), key=lambda x:len(x), reverse=True)
    idx_mapping = {key:idx for idx, key in enumerate(all_pre)}

    mapping = {}
    attr_dic = {}
    for node in travel_nodes:
        new_node =  f"{idx_mapping[node[0]]}_{node[1]}"
        mapping[node] = new_node
        attr_dic[new_node] = node
    renamed_tree = nx.relabel_nodes(fate_tree, mapping)

    nx.set_node_attributes(renamed_tree, attr_dic, 'original')

    return renamed_tree

def add_node_info(fate_tree, ddf, root):

    travel_nodes = list(nx.bfs_tree(fate_tree, root).nodes())

    d_e_dic = {}
    d_pos = {}
    d_u = {}
    d_cumsum = {}
    for node_name in travel_nodes:
        celltype_tuple = node_name[0]
        #print(celltype_tuple)
        tm = node_name[1]

        #if celltype_tuple == (0,1) and tm == 6:
        #    import ipdb
        #    ipdb.set_trace()

        e_lists = [ddf[key][ddf[key]['ubin'] == tm]['edge_idx'] for key in celltype_tuple]
        es = [j for i in e_lists for j in i]
        #if len(es) == 0:
            #import ipdb
            #ipdb.set_trace()
        e_dic = list(Counter(es).items())
        d_e_dic[node_name] = e_dic

        pos_lists = [ddf[key][ddf[key]['ubin'] == tm]['edge_mid_pos'] for key in celltype_tuple]
        poss = [j for i in pos_lists for j in i]
        pos = np.mean(poss, axis=0)
        d_pos[node_name] = pos
        #print(pos)

        u_lists = [ddf[key][ddf[key]['ubin'] == tm]['edge_mid_u'] for key in celltype_tuple]
        us = [j for i in u_lists for j in i]
        u = np.mean(us, axis=0)
        d_u[node_name] = u
        #print(u)

        cumsum_lists = [ddf[key][ddf[key]['ubin'] == tm]['cumsum'] for key in celltype_tuple]
        cumsums = [j for i in cumsum_lists for j in i]
        cumsum = np.mean(cumsums, axis=0)
        d_cumsum[node_name] = cumsum
        #print(node_name,d_u)

    #print(cumsum)
    nx.set_node_attributes(fate_tree, d_e_dic, "ecount")
    nx.set_node_attributes(fate_tree, d_pos, "pos")
    nx.set_node_attributes(fate_tree, d_u, "u")
    nx.set_node_attributes(fate_tree, d_cumsum, "cumsum")
    return fate_tree

def create_branching_tree(pairwise_bdict):
    inv_bdict = {}
    for k, v in pairwise_bdict.items():
        inv_bdict[v] = inv_bdict.get(v, []) + [k]

    keys= set(inv_bdict.keys())
    max_merge = max(keys)

    htree_roots = []
    htree = nx.DiGraph()
    merged_list = merge_common_list(list(inv_bdict[max_merge]))
    for sublist in merged_list:
        for node in sublist:
            htree.add_node((node, ))
            htree.add_edge(tuple(sublist), (node, ))
        htree.nodes[tuple(sublist)]['leaves'] = sublist
        htree.nodes[tuple(sublist)]['time'] = max_merge
        htree_roots.append(tuple(sublist))


    for key in sorted(list(keys-{max_merge}), reverse=True):
        vals = merge_common_list(list(inv_bdict[key]))
        #print(key, vals)
        for val in vals:
            htree, htree_roots = add_branching(key, val, htree, htree_roots)
    root = htree_roots[0]
    return htree, root

def add_branching(tm, val, htree, htree_roots):
    roots = np.array(htree_roots)
    in_roots = np.where(np.array([True if len(set(r)&set(val))>0 else False for r in roots]))[0]

    if len(list(in_roots)) > 0: ## merge > 2 trees
        up_leaves = {j for i in roots[in_roots] for j in i}
        subleaves = set(val) - up_leaves
        all_leaves = tuple(up_leaves | subleaves)
        htree.add_node(tuple(all_leaves))
        htree.nodes[tuple(all_leaves)]['leaves'] = tuple(all_leaves)
        htree.nodes[tuple(all_leaves)]['time'] = tm
        for leaf in subleaves:
            htree.add_edge(tuple(all_leaves), (leaf, ))

        #print('in_roots', in_roots)
        for i in in_roots:
            a_root = roots[i]
            if tuple(a_root) != tuple(all_leaves):
                htree.add_edge(tuple(all_leaves), tuple(a_root))
                htree_roots[i] = tuple(all_leaves)
        htree_roots = list(set(htree_roots))

    else: ### create a new subtree
        #print('val', val)
        for node in val:
            htree.add_node((node, ))
            htree.add_edge(tuple(val), (node, ))
        htree.nodes[tuple(val)]['leaves'] = val
        htree.nodes[tuple(val)]['time'] = tm
        htree_roots.append(tuple(val))
    return htree, htree_roots



def merge_common_list(lst):
    def to_graph(lst):
        G = nx.Graph()
        for part in lst:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(lst):
        """
            treat `l` as a Graph and returns it's edges
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(lst)
        last = next(it)

        for current in it:
            yield last, current
            last = current

    G = to_graph(lst)
    return list(connected_components(G))

def harmonic_trajs_bins(adata: AnnData,
                        graph_name: str = 'X_dm_ddhodge_g_triangulation_circle',
                        evector_name="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                        eigen_n = 20,
                        layout_name: str = 'X_dm_ddhodge_g',
                        full_traj_matrix = 'full_traj_matrix',
                        trajs_clusters = 'trajs_clusters',
                        trajs_use = 100,
                        retain_clusters = [],
                        node_attribute = 'u',
                        bin_number = 10,
                        min_kde_quant_rm = 0.1,
                        kde_sample_n = 1000,
                        random_seed = 2022,
                        ):
    """
    return bins of each trajectory clustere edges
    """
    np.random.seed(random_seed)


    m_full_traj_matrix = adata.uns[full_traj_matrix]
    trajs_use = min(trajs_use, len(m_full_traj_matrix))
    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:eigen_n, :], adata.uns[full_traj_matrix])
    cumsums = list(map(lambda i: [np.cumsum(j) for j in i ], mat_coord_Hspace))


    cluster_list = adata.uns[trajs_clusters]
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    assert(set(retain_clusters).issubset(set(cluster_list)))
    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)


    print(retain_clusters)

    bins_df_dict = {}
    for cluster in tqdm(retain_clusters):
        itrajs = [i for i in np.where(cluster_list == cluster)[0]]
        if len(itrajs) > trajs_use:
            itrajs = np.random.choice(itrajs, trajs_use, replace=False)

        traj_edge_idxs = []
        cumsum_list = []
        for itraj in itrajs:
            traj_mtx = adata.uns[full_traj_matrix][itraj]
            traj_edge_idx = [j for i in np.argmax(np.abs(traj_mtx.astype(int)), axis=0).tolist() for j in i]
            cumsum_list.extend([row for row in np.vstack(cumsums[itraj]).T])
            traj_edge_idxs.extend(traj_edge_idx)

        traj_edge_idxs = np.array(traj_edge_idxs)

        #sampling
        sub_idx = np.random.choice(range(len(traj_edge_idxs)), min(kde_sample_n, len(traj_edge_idxs)), replace=False)
        npcumsum = np.vstack(cumsum_list)[sub_idx, :]
        nptraj_edge_idxs = traj_edge_idxs[sub_idx]
        #print("1  ", nptraj_edge_idxs)
        #kde estimate
        kde = gaussian_kde(nptraj_edge_idxs)(nptraj_edge_idxs)
        kde_keep = np.where(kde > np.quantile(kde, min_kde_quant_rm))[0]
        #print("kde keep", kde_keep.shape)
        #print("cumsum", npcumsum)
        # keep by np.quantile
        #print(nptraj_edge_idxs)
        df_left = pd.DataFrame({'edge_idx': nptraj_edge_idxs[kde_keep]})
        #print(df_left)
        #print(npcumsum[kde_keep])
        df_left['cumsum'] =  list(npcumsum[kde_keep])

        df_left['edge_mid_u'] = df_left['edge_idx'].map(_edge_mid_attribute(adata, graph_name, node_attribute))
        df_left['edge_mid_pos'] = df_left['edge_idx'].map(_edge_mid_points(adata, graph_name, layout_name))

        ## bins by the node attribute rank
        df_left['rank'] = df_left['edge_mid_u'].rank()
        bins = np.linspace(df_left['rank'].min(), df_left['rank'].max(), bin_number+1)
        df_left['bins'] = pd.cut(df_left['rank'], bins , include_lowest=True)
        d_bins = {v:i for  i, v in enumerate(sorted(set(df_left.bins)))}
        df_left["edge_bins"] = df_left['bins'].apply(lambda x: d_bins[x])
        bins_df_dict[cluster] = df_left

    return bins_df_dict



def time_sync_bins(ddf, attr='edge_mid_u', min_bin_number=5):
    maxx, minn, longest_key = max_min_attribute(ddf, attr)
    avg_cut = avg_cut_bins(ddf[longest_key][attr].min(), ddf[longest_key][attr].max(), maxx, minn, min_bin_number)
    for key, df in ddf.items():
        df = u_bins_df(df, avg_cut=avg_cut)
        ddf[key] = df
    return ddf


def pairwise_branching_dict(ddf, bin_attr='ubin', edge_attr='edge_idx', bottom_up=True):
    keys = list(ddf.keys())
    pairs = itertools.combinations(keys, 2)
    pairwise_bdict = {}
    for k1,k2 in pairs:
        df1 = ddf[k1]
        df2 = ddf[k2]
        cosine_list = []
        for i in range(max(df1[bin_attr].max(), df2[bin_attr].max())):
            list1 = df1[df1[bin_attr] == i][edge_attr]
            list2 = df2[df2[bin_attr] == i][edge_attr]
            cosine_list.append(term_frequency_cosine(list1, list2))
        #pairwise_bdict[(k1,k2)] =find_cut_point(cosine_list)
        pairwise_bdict[(k1,k2)] =find_cut_point_bu(cosine_list) if bottom_up else find_cut_point(cosine_list)
    return pairwise_bdict



def pairwise_hbranching_dict(ddf, bin_attr='ubin', cumsum_attr='cumsum', bottom_up=True):
    keys = list(ddf.keys())
    pairs = itertools.combinations(keys, 2)
    pairwise_bdict = {}
    for k1,k2 in pairs:
        df1 = ddf[k1]
        df2 = ddf[k2]
        dist_list = []
        for i in range(max(df1[bin_attr].max(), df2[bin_attr].max())):
            list1 = df1[df1[bin_attr] == i][cumsum_attr]
            list2 = df2[df2[bin_attr] == i][cumsum_attr]
            dist_list.append(norm_distance(list1, list2))
        pairwise_bdict[(k1,k2)] = find_cut_point_bu(dist_list, cut_threshold=1, increase=True) if bottom_up else find_cut_point(dist_list, cut_threshold=1, increase=True)
    return pairwise_bdict



def norm_distance(list1, list2, random_seed=2022, sample_num=100):
    """
    two list of points,
    1. calculate the distance between each sampled point in list1
    1. calculate the distance between each sampled point in list2
    2. get the average as normlized distance in list1 and list2
    """
    np.random.seed(random_seed)


    if len(list1)==0 or len(list2)==0:
        return np.nan

    list1 = np.random.choice(list1, sample_num, replace=True)
    list2 = np.random.choice(list2, sample_num, replace=True)

    # mean distances innner list1 and list2
    l1 = np.vstack(list1)
    l2 = np.vstack(list2)
    dl1 = sklearn.metrics.pairwise_distances(l1)
    dl2 = sklearn.metrics.pairwise_distances(l2)
    dm1 = (dl1.sum() - np.trace(dl1))/(2* np.power(dl1.shape[0], 2)) + 0.00001
    dm2 = (dl2.sum() - np.trace(dl2))/(2* np.power(dl2.shape[0], 2)) + 0.00001

    mean_cumsum_1 = np.mean(list1)
    mean_cumsum_2 = np.mean(list2)

    dist = distance.cdist([mean_cumsum_1]/dm1, [mean_cumsum_2]/dm2)[0][0]

    return dist


def df_attr_counter(ddf, keys, attr='edge_idx', bin_attr='ubin', which_bin=10):
    list_attr = []
    for key in keys:
        df = ddf[key]
        if which_bin in df[bin_attr].unique():
            list_attr.extend([tuple(i) for i in df[df[bin_attr] == which_bin][attr] if isinstance(i, list)])
    return Counter(list_attr)


def max_min_attribute(ddf, attr='edge_mid_u'):
    """
    return max and min of the attribute
    max from the smallest range df
    """
    minn = -1
    maxx = 1e100
    longest = 0
    longest_key = None
    for key, df in ddf.items():
        dfmax = df[attr].max()
        dfmin = df[attr].min()
        minn = max(minn, dfmin)
        maxx = min(maxx, dfmax)
        if dfmax - dfmin > longest:
            longest = dfmax - dfmin
            longest_key = key

    return maxx, minn, longest_key



def avg_cut_bins(dfmin, dfmax, maxx, minn, min_bin_number=5):
    bins_number = int(min_bin_number * (dfmax - dfmin) / (maxx - minn))
    avg_cut = np.linspace(dfmin, dfmax, bins_number+2)[1:-1]
    return avg_cut




def u_bins_df(df, attr='edge_mid_u', min_bin_number=5, avg_cut=None):
    """
    1. calculate the number of bins
    """
    dfmin = df[attr].min()
    dfmax = df[attr].max()
    df.sort_values(by='rank', inplace=True)

    #print(avg_cut)
    avg_cut = list(avg_cut)
    if avg_cut[0] < dfmin:
        avg_cut[0] = dfmin
    for i in range(1, len(avg_cut)):
        #print(i)
        if avg_cut[i] > dfmax:
            avg_cut[i] = dfmax
            # delete avg_cut from index i
            avg_cut = avg_cut[:i+1]
            break


    ## cut the bins
    start_idx = 0
    cut_idx_list = [start_idx,]
    for idx, acut in enumerate(avg_cut):
        start_idx = incremental_avg(df[attr], acut, start_idx=start_idx)
        if idx == 0 and start_idx <=0:
            continue
        if start_idx > 0:
            cut_idx_list.append(start_idx)
        else:
            #cut_idx_list.append(df.shape[0])
            break
    cut_idx_list.append(df.shape[0])

    cut_pairs = pairwise(cut_idx_list)
    df['ubin'] = 0
    df['umean'] = 0
    for idx, (start, end) in enumerate(cut_pairs):
        #print(idx)
        #print(start, end)
        if start == end:
            continue
        #df['ubin'][start:end]= idx
        df.loc[df.index[start:end], 'ubin'] = idx
        #df['umean'][start:end]= df['edge_mid_u'][start:end].mean()
        df.loc[df.index[start:end], 'umean'] = df['edge_mid_u'][start:end].mean()
        #print(bins_number)
    return df




def incremental_avg(ordered_list, reach_value=3, start_idx=0):
    """
    Given an ordered list and start_idx, traverse the list check when the average of top k values is greater than reach_value:
    return index of the cut points
    """
    #https://math.stackexchange.com/questions/106700/incremental-averaging
    # m_n = \frac{1}{n} \sum_{i=1}^n a_i
    # m_n = \frac{(n-1)m_{n-1} + a_n}{n}
    # m_{n-1} = \frac{m_n n - a_n}{n-1}

    ordered_list = list(ordered_list)

    mn = np.average(ordered_list[start_idx:])
    length = len(ordered_list) - start_idx


    for i in reversed(range(start_idx, len(ordered_list))):
        if length==1:
            return i
        mn = (mn * length - ordered_list[i])/(length - 1)
        length -= 1
        #print(i, mn)
        if reach_value >= mn:
            return i
    return -1




def dic_avg_attribute(df, attr='edge_mid_pos', bin_idx='edge_bins'):
    """
    return average position of each bin
    """
    dic = {}
    for i in set(df[bin_idx]):
        dic[i] = np.average(df[df[bin_idx] == i][attr].tolist(), axis=0)
    return dic




def _edge_mid_points(adata: AnnData,
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

def _edge_mid_attribute(adata: AnnData,
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


