import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def graph_layout(adata, graph_name=None, layout='neato', out_name=None, iscopy=False):
    """generate a layout for a graph
    """
    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if iscopy:
        adata = adata.copy()

    layouts = nx.nx_pydot.graphviz_layout(adata.uns[graph_name], prog=layout)

    if out_name:
        adata.obsm[out_name] = np.array([layouts[i] for i in range(len(layouts))])
    else:
        adata.obsm[graph_name] = np.array([layouts[i] for i in range(len(layouts))])

    return adata if iscopy else None
