# Code adapted from https://github.com/kazumits/ddhodge.

## run ddhodge on dimension reduction of adata
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Union
from anndata import AnnData
import scipy
from locale import getpreferredencoding
from scipy.sparse import csr_matrix, issparse
from ..tools.graphconstr import diffusionGraphDM, diffusionGraph

def _graphviz_layout_with_seed(G, prog='neato', random_seed=2022):
    import pydot
    P = nx.nx_pydot.to_pydot(G)
    P.set("start", str(random_seed))
    D_bytes = P.create_dot(prog=prog)
    D = str(D_bytes, encoding=getpreferredencoding())
    if D == "":
        raise ValueError(f"Graphviz layout with {prog} failed")
    Q_list = pydot.graph_from_dot_data(D)
    assert len(Q_list) == 1
    Q = Q_list[0]
    node_pos = {}
    for n in G.nodes():
        str_n = str(n)
        node = Q.get_node(pydot.quote_id_if_necessary(str_n))
        if isinstance(node, list):
            node = node[0]
        pos = node.get_pos()[1:-1]
        if pos is not None:
            xx, yy = pos.split(",")
            node_pos[n] = (float(xx), float(yy))
    return node_pos

##TODO: if the graph is not connected, need try to increase the k to get a connected graph
def ddhodge(
        adata: AnnData,
        basis: str = 'X_pca',
        roots: Union[str, list] = None,
        k: int = 11,
        npc: int = 100,
        ndc: int = 40,
        s: int = 1,
        j: int = 7,
        lmda: float = 1e-4,
        sigma: float = None,
        layout: str = 'neato',
        iscopy: bool = False,
        verbose: bool = True,
        lstsq_method: str = 'lstsq',
        random_seed: int = 2022,
        ):

    """
    ddhodge implementation for dimension reduction.

    Parameters
    ----------
    basis
        Name of the basis to use for dimension reduction if None use normalization from ddhodge to perform pca.
    roots
        Root cells for diffusion graph construction.
    k
        Number of nearest neighbors for graph prunning.
    npc
        Number of principal components for diffusion graph construction.
    ndc
        Number of diffusion components for diffusion graph construction.
    s
        Number of diffusion steps for diffusion graph construction.
    j
        Number of nearest neighbors for diffusion graph construction.
    lmda
        Regularization parameter for edge weights.
    layout
        Graphviz layout to use for visualization, can be one of 'dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo'.

    lstsq_method
        Diffusion psuedo time estimation method, can be one of 'lstsq', 'lsqr', 'lsmr', 'cholesky'.

    random_seed
        Random seed for deterministic eigen decomposition and graph layout (default 2022).
    """
    if basis and basis not in adata.obsm.keys():
        raise ValueError('basis not in adata.obsm.keys()')
    if roots is None:
        raise ValueError('roots is None')

    if iscopy:
        adata = adata.copy()

    if isinstance(roots, str) and roots in adata.obs.keys():
        roots = adata.obs[roots].tolist()
    elif isinstance(roots, list) or isinstance(roots, np.ndarray) or isinstance(roots, pd.Series):
        adata.obs['root'] = roots

    if len(roots) != adata.n_obs:
        raise ValueError('Length of roots is not equal to adata.n_obs')
    if set(roots) != {True, False}:
        raise ValueError('Roots must be boolean and include True and False')

    import warnings
    warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive")
    warnings.filterwarnings("ignore", message="converting matrix of class csr_array to CSC format")

    d = {}
    if basis:
        if isinstance(adata.obsm[basis], pd.DataFrame): ## fixed wrong type
            adata.obsm[basis] = adata.obsm[basis].to_numpy()
        pc = adata.obsm[basis][:, 0:npc]
        d = diffusionGraphDM(pc,roots=roots,k=k,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma, verbose=verbose, lstsq_method=lstsq_method, random_seed=random_seed)
    else:
        d = diffusionGraph(adata.X.T.todense() if scipy.sparse.issparse(adata.X.T) else adata.X.T ,roots=roots,k=k,npc=npc,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma, verbose=verbose, lstsq_method=lstsq_method, random_seed=random_seed)
        adata.obsm['X_pca'] = d['dm']
        basis = 'X_pca'


    adata.uns[f'{basis}_ddhodge_g'] = d['g']
    adata.uns[f'{basis}_ddhodge_A'] = d['A']
    adata.uns[f'{basis}_ddhodge_W'] = d['W']
    adata.uns[f'{basis}_ddhodge_psi'] = d['psi']
    #adata.uns[f'{basis}_ddhodge_phi'] = d['phi']
    adata.uns[f'{basis}_ddhodge_eig'] = d['eig']
    u = [i[1] for i in sorted(nx.get_node_attributes(d['g'], 'u').items(), key=lambda x: x[0])]
    adata.obs['u'] = np.array(u)

    if layout:
        if verbose:
            print(datetime.now(), 'calculate layouts')
        layouts = _graphviz_layout_with_seed(d['g'], prog=layout, random_seed=random_seed)
        adata.obsm[f'{basis}_ddhodge_g'] = np.array([layouts[i] for i in range(len(layouts))])
    if verbose:
        print(datetime.now(), 'done')
    adata.uns["graph_basis"] = f'{basis}_ddhodge_g'
    return adata if iscopy else None
