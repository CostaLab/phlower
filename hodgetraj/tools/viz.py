import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def graph_layout(adata, graph_basis="X_pca", layout='neato', copy=False):
    """generate a layout for a graph
    """

    if copy:
        adata_copy = adata.copy()
    else:
        adata_copy = adata

    graph_name = graph_basis + "_ddhodge_g"

    layouts = nx.nx_pydot.graphviz_layout(adata_copy.uns[graph_name], prog=layout)

    adata_copy.obsm[graph_name + '_' + layout] = np.array([layouts[i] for i in range(len(layouts))])

    return adata_copy if copy else None
