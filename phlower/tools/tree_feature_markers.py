import os
import scipy
import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Union, List


def _edge_two_ends(adata: AnnData,
                   graph_name: str = None,
                   ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    elist = np.array([(x[0], x[1]) for x in adata.uns[graph_name].edges()])
    return {i:v for i, v in enumerate(elist)}

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



def tree_nodes_markers(adata: AnnData,
                       node1: str,
                       nodes2:Union[str, List[str]] = None,
                       iscopy = False,
                       **kwargs):
    """
    use scanpy to find markers
    node1 node2 name all from fate_tree, where ecount stored the edge count
    1. find the cells in node1 and node2 by counting edges two endings
    2. multiple the frequncy of each cell by the edge frequency
    3. construct each array for node1 and node2
    4. create a new AnnData(bdata) for using merged node1 and node2 expression matrix
    5. find markers for node1 and node2 in bdata
    6. shrink the bdata size with only 3 columns
    7. store bdata in adata.uns

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix with cell attributes.
    node1: str
        node1 name from fate_tree
    node2: str
        node2 name from fate_tree

    """
    adata = adata.copy() if iscopy else adata
    if node1 not in adata.uns["fate_tree"].nodes():
        raise ValueError(f"{node1} is not in fate_tree")
    if nodes2 is None:
        raise ValueError(f"nodes2 is None")
    if isinstance(nodes2, str):
        nodes2 = [nodes2]
    for node2 in nodes2:
        if node2 not in adata.uns["fate_tree"].nodes():
            raise ValueError(f"{node2} is not in fate_tree")

    ## construct node1 array
    d_edge2node = _edge_two_ends(adata)
    node1_tuple = adata.uns['fate_tree'].nodes[node1]['ecount']
    d_nodes_node1 = _edgefreq_to_nodefreq(edge_freq=node1_tuple, d_edge2node=d_edge2node)
    X_node1 = adata.X[list(d_nodes_node1.keys()),:]
    if scipy.sparse.issparse(X_node1):
        X_node1 = X_node1.toarray()
    X_node1 = X_node1 * np.array(list(d_nodes_node1.values()))[:, np.newaxis]

    ## construct node2 array
    nodes2_name = nodes2
    if not isinstance(nodes2, str):
        nodes2_name = '.'.join(nodes2)
        nodes2_tuple = []
        for node2 in nodes2:
            node2_tuple = adata.uns['fate_tree'].nodes[node2]['ecount']
            nodes2_tuple.extend(node2_tuple)
    ## construct node2 array
    else:
        nodes2_tuple = adata.uns['fate_tree'].nodes[nodes2]['ecount']

    d_nodes_nodes2 = _edgefreq_to_nodefreq(edge_freq=nodes2_tuple, d_edge2node=d_edge2node)
    X_nodes2 = adata.X[list(d_nodes_nodes2.keys()),:]
    if scipy.sparse.issparse(X_nodes2):
        X_nodes2 = X_nodes2.toarray()
    X_nodes2 = X_nodes2 * np.array(list(d_nodes_nodes2.values()))[:, np.newaxis]



    # concat and initial a new anndata
    group = [node1] * X_node1.shape[0]  + [nodes2_name]*X_nodes2.shape[0]
    X_merge = pd.DataFrame(np.concatenate((X_node1, X_nodes2)))
    X_merge.columns = adata.var_names
    X_merge.index =  range(X_merge.shape[0])
    ## normalize columns to 1 avoid too large value
    normalized_X=(X_merge-X_merge.min())/(X_merge.max()-X_merge.min())


    bdata = sc.AnnData(normalized_X)
    bdata.obs['compare'] = group

    # find markers using scanpy
    sc.tl.rank_genes_groups(bdata, groupby='compare', groups=[node1], reference=nodes2_name, **kwargs)
    # shrink the helping anndata object
    bdata = bdata[:, :3]## make it smaller for storage.
    # save the result to adata
    adata.uns[f'markers_{node1}_vs_{nodes2_name}'] = bdata

    return adata if iscopy else None
