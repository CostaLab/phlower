import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def graph_layout(adata, graph_name="X_pca_ddhodge_g", layout='neato', out_name=None, copy=False):
    """generate a layout for a graph
    """

    if copy:
        adata = adata.copy()

    layouts = nx.nx_pydot.graphviz_layout(adata.uns[graph_name], prog=layout)

    if not out_name:
        adata.obsm[out_name] = np.array([layouts[i] for i in range(len(layouts))])
    else:
        adata.obsm[graph_name] = np.array([layouts[i] for i in range(len(layouts))])

    return adata if copy else None
