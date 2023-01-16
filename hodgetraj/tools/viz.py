import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def graph_layout(adata, graph_name="X_pca_ddhodge_g", layout='neato', out_name=None, copy=False):
    """generate a layout for a graph
    """

    if copy:
        adata_copy = adata.copy()
    else:
        adata_copy = adata


    layouts = nx.nx_pydot.graphviz_layout(adata_copy.uns[graph_name], prog=layout)

    if not out_name:
        adata_copy.obsm[out_name] = np.array([layouts[i] for i in range(len(layouts))])
    else:
        adata_copy.obsm[graph_name] = np.array([layouts[i] for i in range(len(layouts))])

    return adata_copy if copy else None
