import itertools
import numpy as np
import colorcet as cc
import pandas as pd
import gudhi as gh
import matplotlib.pyplot as plt
from anndata import AnnData
from itertools import chain
from collections import defaultdict
from typing import Iterable, List, Union, Optional, Set, Tuple, TypeVar
import gudhi as gd

def persisitence_barcodes(adata: AnnData,
                          include_holes:bool=True,
                          persistence:str='persistence',
                          barcodes_dim:List[int]=[0],
                          min_persistence:float=0.01,
                          show_threshold:bool=True,
                          show_legend:bool=True,
                          manual_threshold:Optional[List[float]] = None,
                          ax=None,
                          **args):
    """
    Plot the simplex tree barcodes

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    persistence: str
        name in adata.uns
    """
    if persistence not in adata.uns:
        raise ValueError(f"persistence: {persistence} not in adata.uns, please check!")
    if not set(barcodes_dim) <= set([0, 1, 2, 3]):
        raise ValueError(f"barcodes_dim: {barcodes_dim} should be in [0, 1, 2, 3]")


    if include_holes:
        simplex_tree = adata.uns[persistence]['simplextree_tri_ae']
    else:
        simplex_tree = adata.uns[persistence]['simplextree_tri']

    pers = simplex_tree.persistence(min_persistence=min_persistence)
    barcode = [b for b in pers if b[0] in barcodes_dim]
    ax = gd.plot_persistence_barcode(barcode, axes=ax, **args)
    limit = max(adata.uns[persistence]['filter_num']+10, ax.get_xlim()[1])
    ax.set_xlim(0, limit)
    if show_threshold:
        if not manual_threshold:
            ax.axvline(x=adata.uns[persistence]['filter_num'], ls='--', color='blue')
        else:
            for th in manual_threshold:
                ax.axvline(x=th, ls='--', color='blue')
    if not show_legend:
        ax.get_legend().remove()

    return ax


def persisitence_birth_death(adata: AnnData,
                             include_holes:bool=True,
                             persistence:str='persistence',
                             min_persistence:float=0.01,
                             show_threshold:bool=True,
                             show_legend:bool=True,
                             manual_threshold:Optional[List[float]] = None,
                             ax=None,
                             **args):
    """
    Plot the persistence birth death diagram

    Parameters
    ----------
    adata: AnnData
        an Annodata object
    persistence: str
        name in adata.uns
    """
    if persistence not in adata.uns:
        raise ValueError(f"persistence: {persistence} not in adata.uns, please check!")

    if include_holes:
        simplex_tree = adata.uns[persistence]['simplextree_tri_ae']
    else:
        simplex_tree = adata.uns[persistence]['simplextree_tri']

    pers = simplex_tree.persistence(min_persistence=min_persistence)
    ax = gd.plot_persistence_diagram(pers, axes=ax, **args)
    if show_threshold:
        if not manual_threshold:
            ax.axhline(y=adata.uns[persistence]['filter_num'], ls='--', color='blue')
        else:
            for th in manual_threshold:
                ax.axhline(y=th, ls='--', color='blue')
    if not show_legend:
        ax.get_legend().remove()
    return ax
