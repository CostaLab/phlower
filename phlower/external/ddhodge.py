# Code adapted from https://github.com/kazumits/ddhodge.

## run ddhodge on dimension reduction of adata
import numpy as np
import pandas as pd
import networkx as nx
from typing import Union
from anndata import AnnData
import scipy
from scipy.sparse import csr_matrix, issparse
from ..tools.graphconstr import diffusionGraphDM, diffusionGraph

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

    d = {}
    if basis:
        pc = adata.obsm[basis].iloc[:, 0:npc]
        d = diffusionGraphDM(pc,roots=roots,k=k,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma)
    else:
        d = diffusionGraph(adata.X.T.todense() if scipy.sparse.issparse(adata.X.T) else adata.X.T ,roots=roots,k=k,npc=npc,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma)
        adata.obsm['X_dm'] = d['dm']
        basis = 'X_dm'


    adata.uns[f'{basis}_ddhodge_g'] = d['g']
    adata.uns[f'{basis}_ddhodge_A'] = d['A']
    adata.uns[f'{basis}_ddhodge_W'] = d['W']
    adata.uns[f'{basis}_ddhodge_psi'] = d['psi']
    adata.uns[f'{basis}_ddhodge_phi'] = d['phi']
    adata.uns[f'{basis}_ddhodge_eig'] = d['eig']
    if layout:
        print('calculate layouts')
        layouts = nx.nx_pydot.graphviz_layout(d['g'], prog=layout)
        adata.obsm[f'{basis}_ddhodge_g'] = np.array([layouts[i] for i in range(len(layouts))])
    print('done')

    return adata if iscopy else None