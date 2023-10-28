import re
import warnings
import networkx as nx
from copy import deepcopy
from io import StringIO
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, nbinom, pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from matplotlib import font_manager
import matplotlib.patheffects as pe

from .plotting import nxdraw_group

## adjusted from: https://github.com/kunwang34/PhyloVelo
def fate_velocity_plot_cumsum(
    adata,
    fate_tree='fate_tree',
    layout_name='cumsum',
    group_name ="group",
    graph_name=None,
    figtype: "str:stream, grid, point" = "grid",
    #nn: "str:knn, radius" = "radius",
    grid_density: int = 50,
    streamdensity: int = 1,
    n_neighbors: int = 4,
    embedd_label=True,
    embedd_label_style="text",
    embedd_label_font=10,
    show_legend: bool = False,
    show_cumsum= True,
    radius: float = 0.5,
    ax=None,
    node_alpha: float = 0.5,
    maxlinewidth = 5,
    lines_sample_ratio=0.3,
    alpha: float = 0.5,
    #streamdensity: float = 1.5,

    **kwargs
):
    """
    Plot fate velocity on a fate tree.

    Parameters
    ----------
    adata: anndata.AnnData
        Annotated data matrix.
    fate_tree: str
        The name of the fate tree in adata.uns.
    layout_name: str
        The name of the layout in adata.uns.
    group_name: str
        The name of the group in adata.uns.
    graph_name: str
        The name of the graph in adata.uns.
    figtype: str
        The type of the figure. "grid" for grid plot, "stream" for stream plot, "point" for point plot.
    grid_density: int
        The density of the grid.
    streamdensity: float
        The density of the stream.
    n_neighbors: int
        The number of neighbors for the nearest neighbor graph.
    embedd_label: bool
        Whether to show the label of the embedding.
    embedd_label_style: str
        The style of the label of the embedding. "text" for text label, "arrow" for arrow label.
    embedd_label_font: int
        The font size of the label of the embedding.
    show_legend: bool
        Whether to show the legend.
    show_cumsum: bool
        Whether to show the cumulative sum of the fate probability.
    radius: float
        The radius of the grid.
    ax: matplotlib.axes.Axes
        The axes of the plot.
    node_alpha: float
        The transparency of the nodes.
    maxlinewidth: float
        The maximum width of the lines.
    lines_sample_ratio: float
        The ratio of the lines to be sampled.
    alpha: float
        The transparency of the lines.
    """

    if group_name not in adata.obs:
        raise Exception("group name not found")

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if fate_tree not in adata.uns:
        raise Exception("fate tree not found")

    ax = plt.gca() if ax is None else ax
    if layout_name == "cumsum":
        grid_length = grid_diagonal_length_cumsum(adata.uns[fate_tree], grid_density=grid_density)
    else:
        grid_length = grid_diagonal_length(adata.obsm[layout_name], grid_density)
    bin_umap, bin_velocity = bin_umap_velocity(adata, fate_tree, layout_name, grid_length=grid_length, radius=radius)
    coor_velocity_plot(bin_umap,
                       bin_velocity,
                       ax=ax,
                       grid_density=grid_density,
                       figtype=figtype,
                       streamdensity=streamdensity,
                       n_neighbors=n_neighbors,
                       maxlinewidth = maxlinewidth,
                       radius=radius,
                       alpha=alpha,
                       **kwargs)
    if show_cumsum:
        from .plotting import plot_trajectory_harmonic_lines
        plot_trajectory_harmonic_lines(adata, sample_ratio=lines_sample_ratio, show_legend=False, ax=ax)

#endf fate_velocity_plot


def fate_velocity_plot(
    adata,
    fate_tree='fate_tree',
    layout_name=None,
    group_name ="group",
    graph_name=None,
    figtype: "str:stream, grid, point" = "grid",
    #nn: "str:knn, radius" = "radius",
    grid_density: int = 50,
    streamdensity: int = 1.2,
    n_neighbors: int = 4,
    embedd_label=True,
    embedd_label_style="text",
    embedd_label_font=10,
    show_legend: bool = False,
    show_nodes = True,
    radius: float = 50,
    ax=None,
    node_alpha: float = 0.5,
    maxlinewidth = 4,
    alpha: float = 0.5,
    #streamdensity: float = 1.5,

    **kwargs
):

    """
    Plot fate velocity on a fate tree.

    Parameters
    ----------
    adata: anndata.AnnData
        Annotated data matrix.
    fate_tree: str
        The name of the fate tree in adata.uns.
    layout_name: str
        The name of the layout in adata.uns.
    group_name: str
        The name of the group in adata.uns.
    graph_name: str
        The name of the graph in adata.uns.
    figtype: str
        The type of the figure. "grid" for grid plot, "stream" for stream plot, "point" for point plot.
    grid_density: int
        The density of the grid.
    streamdensity: float
        The density of the stream.
    n_neighbors: int
        The number of neighbors for the nearest neighbor graph.
    embedd_label: bool
        Whether to show the label of the embedding.
    embedd_label_style: str
        The style of the label of the embedding. "text" for text label, "arrow" for arrow label.
    embedd_label_font: int
        The font size of the label of the embedding.
    show_legend: bool
        Whether to show the legend.
    radius: float
        The radius of the grid.
    ax: matplotlib.axes.Axes
        The axes of the plot.
    node_alpha: float
        The transparency of the nodes.
    maxlinewidth: float
        The maximum width of the lines.
    alpha: float
        The transparency of the lines.
    """
    if group_name not in adata.obs:
        raise Exception("group name not found")

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if fate_tree not in adata.uns:
        raise Exception("fate tree not found")

    ax = plt.gca() if ax is None else ax
    if layout_name == "cumsum":
        grid_length = grid_diagonal_length_cumsum(adata.uns[fate_tree], grid_density=grid_density)
    else:
        grid_length = grid_diagonal_length(adata.obsm[layout_name], grid_density)
    bin_umap, bin_velocity = bin_umap_velocity(adata, fate_tree, layout_name, grid_length=grid_length, radius=radius)
    coor_velocity_plot(bin_umap,
                       bin_velocity,
                       ax=ax,
                       grid_density=grid_density,
                       figtype=figtype,
                       streamdensity=streamdensity,
                       n_neighbors=n_neighbors,
                       maxlinewidth = maxlinewidth,
                       radius=radius,
                       alpha=alpha,
                       **kwargs)
    if show_nodes:
        nxdraw_group(adata,
                     group_name=group_name,
                     ax=ax,
                     layout_name=layout_name,
                     graph_name=graph_name,
                     node_size=4,
                     label = embedd_label,
                     labelsize = embedd_label_font,
                     labelstyle = embedd_label_style,
                     show_edges=False,
                     show_legend=show_legend,
                     alpha=node_alpha,
                     )



def grid_diagonal_length(umap, grid_density=20):
    x_min, x_max = umap[:, 0].min(), umap[:, 0].max()
    y_min, y_max = umap[:, 1].min(), umap[:, 1].max()
    x_len = x_max - x_min
    y_len = y_max - y_min
    grid_len = max(x_len, y_len) / grid_density
    return np.sqrt(2)*grid_len

def grid_diagonal_length_cumsum(fate_tree, grid_density=20):

    cumsum = nx.get_node_attributes(fate_tree, 'cumsum')

    x_values = np.array([i[0] for i in cumsum.values() if len(i)>=2])
    y_values = np.array([i[1] for i in cumsum.values() if len(i)>=2])
    #print(x_values.min(), x_values.max())
    #print(y_values.min(), y_values.max())


    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    x_len = x_max - x_min
    y_len = y_max - y_min
    grid_len = max(x_len, y_len) / grid_density
    return np.sqrt(2)*grid_len



def bin_umap_velocity(adata,
                      fate_tree='fate_tree',
                      layout_name=None,
                      grid_length=1,
                      radius = 50,
                      ):

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if fate_tree not in adata.uns:
        raise Exception("fate tree not found")
    bin_umap = []
    bin_velocity = []
    for a,e in adata.uns[fate_tree].edges():
        if layout_name != 'cumsum':
            xa= adata.uns['fate_tree'].nodes[a][layout_name]
            xe= adata.uns['fate_tree'].nodes[e][layout_name]
        else:
            #print("a,e:", a,e)
            if a == "root" or e == "root":
                continue
            xa= adata.uns['fate_tree'].nodes[a]['cumsum'][:2]
            xe= adata.uns['fate_tree'].nodes[e]['cumsum'][:2]

        #print("xa, xe:", xa, xe, xa-xe, grid_length)
        if np.linalg.norm(xa-xe)*3 > grid_length:
            n = int(np.linalg.norm(xa-xe)*radius*2/grid_length)
            for i in range(n):
                bin_umap.append(xa + (xe-xa)*i/n)
                bin_velocity.append(xe-xa)
        else:
            bin_umap.append(xa)
            bin_velocity.append(xe-xa)
    return np.array(bin_umap), np.array(bin_velocity)
#endf

def Coor_velocity_embedding_to_grid(bin_umap:np.array,
                            bin_vel: np.array,
                            grid_density:int=20,
                            n_neighbors:int=4,
                            radius:float=100,
                            ):

    """
    bin_umap: np.array, bin ddf start x,y coordinate
    bin_vel: np.array, bin ddf x_e-x, y_e-x

    """
    def generate_grid(xlim=(-1, 1), ylim=(-1, 1), density: int = 20):
        Xg, Yg = np.mgrid[
            xlim[0] : xlim[1] : density * 1j, ylim[0] : ylim[1] : density * 1j
        ]
        grid = []
        for i in range(density):
            for j in range(density):
                grid.append([Xg[i][j], Yg[i][j]])
        grid = np.array(grid)
        return Xg, Yg, grid
    def get_weight(x:list, distance:list, scale, length: int):
        oh = [0] * length
        distance = list(distance)
        for i in x:
            oh[i] = scale - distance.pop(0)
        return np.array(oh)

    umap = np.array(bin_umap)

    xrange = max(umap[:, 0]) - min(umap[:, 0])
    yrange = max(umap[:, 1]) - min(umap[:, 1])
    Xg, Yg, grid = generate_grid(
        xlim=(min(umap[:, 0]) - 0.05 * xrange, max(umap[:, 0]) + 0.05 * xrange),
        ylim=(min(umap[:, 1]) - 0.05 * yrange, max(umap[:, 1]) + 0.05 * yrange),
        density=grid_density,
    )
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius).fit(umap)
    distances, indices = nbrs.radius_neighbors(grid)
    vel = bin_vel
    vel_grid = []
    for i, d in zip(indices, distances):
        #print("i:d", i, d)
        if len(i) > 3:
            vel_grid.append(get_weight(i, d, radius, vel.shape[0]).dot(vel) / len(i))
        else:
            vel_grid.append(np.zeros(2))

    vel_grid = np.array(vel_grid)
    lengths = np.sqrt((vel_grid**2).sum(0))
    linewidth = 2 * lengths / lengths[~np.isnan(lengths)].max()
    Ug = vel_grid[:, 0].reshape(grid_density, grid_density)
    Vg = vel_grid[:, 1].reshape(grid_density, grid_density)
    return Xg,Yg,Ug,Vg
#endf velocity_embedding_to_grid

def coor_velocity_plot(
    pts,
    vel,
    ax,
    figtype: "str:stream, grid, point" = "grid",
    #nn: "str:knn, radius" = "radius",
    grid_density: int = 20,
    n_neighbors: int = 4,
    streamdensity: float = 1.5,
    radius: float = 2,
    alpha=0.5,
    maxlinewidth=4,
    **kwargs
):
    '''
    Project velocities into embedding

    Args:
        pts:
            UMAP/tSNE coordinates
        vel:
            Velocity vector
        ax:
            matplotlib.axes
        figtype:
            'stream', 'grid' or 'point'(single cell)
        nn:
            knn or radius neighbors to use
        grid_density:
            density of the grid
        n_neighbors:
            How much neighbors, works when nn=='knn'
        radius:
            How large radius, works when nn='radius'
        streamdensity:
            Density of streamplot, works when figtype==stream
        xlim:
            Grid bound on x axis
        ylim:
            Grid bound on y axis

    Return:
        matplotlib.axes

    '''
    x = pts

    headwidth = kwargs.pop("headwidth", 3)
    headlength = kwargs.pop("headlength", 2)

    if figtype == "point":
        ax.quiver(
            x[:, 0],
            x[:, 1],
            vel[:, 0],
            vel[:, 1],
            headwidth=headwidth,
            headlength=headlength,
            alpha=alpha,
        )
        return ax
    Xg, Yg, Ug, Vg = Coor_velocity_embedding_to_grid(
        pts, vel,  grid_density, n_neighbors, radius
    )

    if figtype == "grid":
        ax.quiver(Xg, Yg, Ug, Vg, zorder=3, headwidth=headwidth, headlength=headlength, alpha=alpha)

    if figtype == "stream":

        lw_coef = kwargs.pop("lw_coef", 1)
        linewidth = kwargs.pop("linewidth", lw_coef * np.sqrt(Ug.T**2 + Vg.T**2))
        linewidth = np.sqrt(maxlinewidth*(linewidth-min(linewidth[~np.isnan(linewidth)]))/(max(linewidth[~np.isnan(linewidth)])-min(linewidth[~np.isnan(linewidth)])))
        #import skimpy

        #print(skimpy.skim(pd.DataFrame(list(linewidth.ravel()))))
        arrowsize = kwargs.pop("arrowsize", 1)
        minlength = kwargs.pop("minlength", 0.1)
        maxlength = kwargs.pop("maxlength", 4)
        color = kwargs.pop("color", "k")
        cmap = kwargs.pop("cmap", None)

        ax.streamplot(
            Xg.T,
            Yg.T,
            Ug.T,
            Vg.T,
            linewidth=linewidth,
            density=streamdensity,
            color=color,
            zorder=3,
            arrowsize=arrowsize,
            minlength=minlength,
            maxlength=maxlength,
            cmap=cmap,
        )
    return ax
#endf
