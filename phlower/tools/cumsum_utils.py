import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from anndata import AnnData
from datetime import datetime
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from typing import Union, Optional, Tuple, Sequence, Dict, Any

from ..util import find_knee
from .tree_utils import _edge_two_ends
from .trajectory import cumsum_Hspace, M_create_matrix_coordinates_trajectory_Hspace


def edge_cumsum_median(adata,
                       full_traj_matrix:str="full_traj_matrix",
                       clusters:str = "trajs_clusters",
                       evector_name:str = None,
                       verbose=True,
                      ):
    """
    for all edges in the cumsum space, we calculate the median of the edge cumsum coordinate

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    full_traj_matrix: `str` (default: `full_traj_matrix`)
        the key in `adata.uns` where the full trajectory matrix is stored
    clusters: `str` (default: `trajs_clusters`)
        the key in `adata.uns` where the cluster labels are stored
    evector_name: `str` (default: None)
        the key in `adata.uns` where the eigenvectors are stored
    verbose: `bool` (default: `True`)
        whether to print the progress bar
    """
    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"
    if 'eigen_value_knee' in adata.uns.keys():
        knee = adata.uns['eigen_value_knee']
    else:
        knee = find_knee(adata)
    if verbose:
        print(datetime.now(), "projecting trajs to harmonic...")
    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][:knee],
                                                                                adata.uns[full_traj_matrix])
    if verbose:
        print(datetime.now(), "cumsum...")
    cumsums =cumsum_Hspace(mat_coord_Hspace, range(knee))


    if verbose:
        print(datetime.now(), "edge cumsum dict...")

    itrajs = range(len(cumsums))
    edges_cumsum_dict = defaultdict(list)
    for itraj in tqdm(itrajs, desc="edges projection"):
        traj_mtx = adata.uns[full_traj_matrix][itraj]
        traj_edge_idx = [j for i in np.argmax(np.abs(traj_mtx.astype(int)), axis=0).tolist() for j in i]
        for i, edge_idx in enumerate(traj_edge_idx):
            edges_cumsum_dict[edge_idx].append(cumsums[itraj][i, ])
    edge_median_dict = {k: np.median(v, axis=0) for k,v in edges_cumsum_dict.items()}
    #edge_median_dict = {k: np.mean(v, axis=0) for k,v in edges_cumsum_dict.items()}
    if verbose:
        print(datetime.now(), "done...")
    return edge_median_dict
#endf edge_cumsum_median

def node_cumsum_coor(adata, d_edge, d_e2n, approximate_k=5, pca_name='X_pca'):
    """
    construct node to edges list dict to enumerate all edge connect to this node
    When a node is not visited, just find the top 5 nearest nodes has been visited and use their average of

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    d_edge: `dict`
        the edge to its median cumsum coordinate dict
    d_e2n: `dict`
        the edge to its two ends node dict
    approximate_k: `int` (default: 5)
        when a node has no edge being visited, find its 5 nearest visited nodes and assign the average the cumsum coordinate
    """
    d_n2e = defaultdict(list)
    for k, vs in d_e2n.items():
        for v in vs:
            d_n2e[v].append(k)
    ## get mean of a node by the edge cumsum coordinate
    d_cumsum_nodes = {}
    unvisited = []
    for k,vs in d_n2e.items():
        cumsum_arr = np.array([d_edge[v] for v in vs if v in d_edge])
        if len(cumsum_arr) == 0:
            unvisited.append(k)
            continue
        d_cumsum_nodes[k] = np.mean(cumsum_arr, axis=0)


    if len(unvisited) > 0:
        print(f"there are {len(unvisited)} nodes not visited in the tree")
        print(f"will approximate the node cumsum coordinate by {approximate_k} nearest accessible neighbors")
    ## assign the mean cumsum coordinate to the missing node.
    X = adata.obsm[pca_name]
    nbrs = NearestNeighbors(n_neighbors=len(unvisited)+approximate_k + 1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    s = set(unvisited)
    for uv in unvisited:
        ids = [i for i in indices[uv][1:] if i not in s][:approximate_k]
        cumsum_arr = np.array([d_cumsum_nodes[v] for v in ids])
        d_cumsum_nodes[uv] = np.mean(cumsum_arr, axis=0)

    return d_cumsum_nodes
#endf node_cumsum_coor

def add_root_cumsum(adata, evector_name=None, fate_tree:str='fate_tree', iscopy=False):
    """
    root cumsum is the mean of the root node

    Since the random walk is a wide range start, we need the root be the very first nodes in the tree
    The cumsum would be not available for the root node.

    Thus the root cumsum is just can be the harmonic coordinate of the root nodes.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    evector_name: `str` (default: None)
        the key in `adata.uns` where the eigenvectors are stored
    fate_tree: `str` (default: `fate_tree`)
        the key in `adata.uns` where the fate tree is stored
    iscopy: `bool` (default: `False`)
        whether to return a copy of adata or not
    """
    import networkx as nx

    adata = adata.copy() if iscopy else adata

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"
    if 'eigen_value_knee' in adata.uns.keys():
        knee = adata.uns['eigen_value_knee']
    else:
        knee = phlower.tl.find_knee(adata)

    n_edges = adata.uns[evector_name].shape[1]
    n_edges


    n_ecount = len(adata.uns[fate_tree].nodes['root']['ecount'] )
    n_ecount

    m = np.zeros(shape=(n_edges, n_ecount))
    for i, (k,v) in enumerate(adata.uns[fate_tree].nodes['root']['ecount'][:knee]):
        m[k, i] = 1
    coord_Hspace = adata.uns[evector_name][:knee] @ m
    attrs = {'root': {"cumsum": np.mean(coord_Hspace, axis=1)}}
    #print(attrs)
    nx.set_node_attributes(adata.uns[fate_tree], attrs)

    return adata if iscopy else None
#endf add_root_cumsum

def trans_tree_node_attr(adata, from_='fate_tree', to_='stream_tree', attr='cumsum', iscopy=False):
    """
    transfer node attribute from fate_tree to stream_tree

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    from_: `str` (default: `fate_tree`)
        the key in `adata.uns` where the node attribute is stored
    to_: `str` (default: `stream_tree`)
        the key in `adata.uns` where the node attribute is stored
    attr: `str` (default: `cumsum`)
        the node attribute to be transfered
    iscopy: `bool` (default: `False`)
        whether to return a copy of adata or not
    """
    adata = adata.copy() if iscopy else adata
    if attr not in adata.uns[from_].nodes['root']:
        raise ValueError(f"attr {attr} is not an node attribute of tree {from_}")
    attrs = nx.get_node_attributes(adata.uns[from_], attr)
    to_attrs = {k:{attr: attrs[k]} for k in adata.uns[to_].nodes() if k in attrs}
    nx.set_node_attributes(adata.uns[to_], to_attrs)

    return adata if iscopy else None
#endf trans_tree_node_attr


def node_cumsum_mean(adata,
                     graph_name = None,
                     full_traj_matrix:str="full_traj_matrix",
                     clusters:str = "trajs_clusters",
                     evector_name:str = None,
                     approximate_k:int = 5,
                     cumsum_name:str = "cumsum",
                     pca_name:str = "X_pca",
                     iscopy=False,
                     verbose=True,
                     ):

    """
    get each node cumsum coordinate:
        1. get each edge cumsum coordinate median
        2. average cumsum of all edges that connect to a node
        3. add the cumsum to cumsum_mean

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object
    graph_name: `str` (default: None)
        the key in `adata.uns` where the graph is stored
    full_traj_matrix: `str` (default: `full_traj_matrix`)
        the key in `adata.uns` where the full trajectory matrix is stored
    clusters: `str` (default: `trajs_clusters`)
        the key in `adata.uns` where the cluster labels are stored
    evector_name: `str` (default: None)
        the key in `adata.uns` where the eigenvectors are stored
    approximate_k: `int` (default: 5)
        the number of nearest neighbors to approximate the cumsum coordinate of the missing node
    iscopy: `bool` (default: `False`)
        whether to return a copy of adata or not
    verbose: `bool` (default: `True`)
        whether to print the progress bar or not
    """
    adata = adata.copy() if iscopy else adata

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if len(adata.uns['fate_tree'].nodes['root']['cumsum']) == 0:
            add_root_cumsum(adata, evector_name=evector_name, fate_tree='fate_tree')
    ## transfer when stream has been created
    if 'stream_tree' in adata.uns.keys():
        trans_tree_node_attr(adata, from_='fate_tree', to_='stream_tree',  attr='cumsum')

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    d = edge_cumsum_median(adata, full_traj_matrix, clusters, evector_name, verbose)
    d_e2n = _edge_two_ends(adata, graph_name=graph_name)
    d_node_cumsum = node_cumsum_coor(adata, d, d_e2n, approximate_k=approximate_k, pca_name=pca_name)
    assert(len(d_node_cumsum) == adata.n_obs)

    ## sort by the node order of adata
    cumsum_mean = np.array([j for i,j in sorted(d_node_cumsum.items(), key=lambda x: x[0], reverse=False)])
    adata.obsm[cumsum_name] = cumsum_mean
    return adata if iscopy else None
#endf node_cumsum_mean
