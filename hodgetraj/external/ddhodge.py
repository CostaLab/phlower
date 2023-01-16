# Code adapted from https://github.com/kazumits/ddhodge.

## run ddhodge on dimension reduction of adata
import numpy as np
import pandas as pd
from typing import Union
from anndata import AnnData
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
        copy: bool = False,
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

    if copy:
        adata_copy = adata.copy()
    else:
        adata_copy = adata

    if isinstance(roots, str):
        roots = adata_copy.obs[roots].tolist()
    elif isinstance(roots, list) or isinstance(roots, np.ndarray) or isinstance(roots, pd.Series):
        adata_copy.obs['roots'] = np.array(roots).tolist()

    d = {}
    if basis:
        pc = adata_copy.obsm[basis][:, 0:npc]
        d = diffusionGraphDM(pc,roots==roots,k=k,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma)
    else:
        d = diffusionGraph(adata_copy.X.T,roots==roots,k=k,npc=npc,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma)
        adata_copy.obsm['X_dm'] = d['dm']
        basis = 'X_dm'


    adata_copy.uns[f'{basis}_ddhodge_g'] = d['g']
    adata_copy.uns[f'{basis}_ddhodge_A'] = d['A']
    adata_copy.uns[f'{basis}_ddhodge_W'] = d['W']
    adata_copy.uns[f'{basis}_ddhodge_psi'] = d['psi']
    adata_copy.uns[f'{basis}_ddhodge_phi'] = d['phi']
    adata_copy.uns[f'{basis}_ddhodge_eig'] = d['eig']

    return adata if copy else None
