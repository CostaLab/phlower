from anndata import AnnData
from typing import Union, Optional, Sequence, Tuple, Mapping, Iterable, Callable
import numpy as np
import pandas as pd
import networkx as nx
from ..util import networkx_node_to_df


def _edgefreq_to_nodefreq(edge_freq, d_edge2node):
    """
    from edge frequency to node frequency
    for each edge, it has two nodes, the node frequency is the sum of edge frequency
    """
    from collections import defaultdict
    d_node_freq = defaultdict(int)
    for x, y in edge_freq:
        #print(x,y)
        a,b = d_edge2node.get(x, None)
        d_node_freq[a] += y
        d_node_freq[b] += y

    return d_node_freq


def _edge_two_ends(adata: AnnData,
                   graph_name: str = None,
                   ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    elist = np.array([(x[0], x[1]) for x in adata.uns[graph_name].edges()])
    return {i:v for i, v in enumerate(elist)}

def print_stream_labels(adata, tree='stream_tree', attr='original'):
    import functools
    import networkx as nx
    arr_tuple = np.fromiter(nx.get_node_attributes(adata.uns[tree], attr).values(), tuple)
    return functools.reduce(lambda a,b: set(a)|set(b), arr_tuple)


def change_stream_labels(adata, tree='stream_tree', attr='original', from_to_dict={}, iscopy=False):
    """
    Change the labels of the stream tree nodes.
    When annotation is changed, need to change the attribute of the tree nodes.
    """
    adata = adata.copy() if iscopy else adata
    nodes = adata.uns[tree].nodes()
    print("changing:", from_to_dict)
    for node in nodes:
        assert(isinstance(adata.uns[tree].nodes[node][attr], tuple))
        adata.uns[tree].nodes[node][attr] = tuple([from_to_dict.get(i, i) for i in adata.uns[tree].nodes[node][attr]])
    return adata if iscopy else None



def get_minimum_nodes(adata, tree='fate_tree', name='4_67'):
    """
    for fate_tree, each branch has a series of nodes increasingly ordered,
    the minimum nodes are the first nodes of each branch
    """

    if name =='root':
        return 'root'

    nodes = adata.uns[tree].nodes()
    predix = name.split('_')[0]
    nodes = [node for node in nodes if node.startswith(predix)]
    appendix = min([int(node.split('_')[1]) for node in nodes])
    return predix + '_' + str(appendix)


def get_rank_genes_group(adata, name):
    """
    helping function to get the rank genes of a group from the fate_tree differential expression
    """
    if name not in adata.uns.keys():
        raise ValueError("name not in adata.uns.keys()")
    return adata.uns[name].uns['rank_genes_groups']

def get_markers_df(adata, name):
    rank_groups = adata.uns[name].uns['rank_genes_groups']
    df = pd.DataFrame({"names" : [i[0] for i in rank_groups['names'].tolist()],
                        "scores" : [i[0] for i in rank_groups['scores'].tolist()],
                        "pvals"  : [i[0] for i in rank_groups['pvals'].tolist()],
                        "pvals_adj" : [i[0] for i in rank_groups['pvals_adj'].tolist()],
                        "logfoldchanges" : [i[0] for i in rank_groups['logfoldchanges'].tolist()], }
                      )
    return df

def fate_tree_full_dataframe(adata, tree='fate_tree', graph_name=None):
    dff = networkx_node_to_df(adata.uns[tree])
    d_edge2node = _edge_two_ends(adata)
    dff['ncount'] = dff['ecount'].apply(lambda x: list(_edgefreq_to_nodefreq(x, d_edge2node).items()))
    return dff




