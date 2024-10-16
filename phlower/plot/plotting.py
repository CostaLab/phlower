import itertools
import numpy as np
import colorcet as cc
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from itertools import chain
from collections import defaultdict
from typing import Iterable, List, Union, Optional, Set, Tuple, TypeVar
from ..tools.trajectory import M_create_matrix_coordinates_trajectory_Hspace

from ..util import get_uniform_multiplication, kde_eastimate, norm01, module_check_install
from ..tools.harmonic_pseudo_tree import get_nodes_celltype_counts

V = TypeVar('V')
def edges_on_path(path: List[V]) -> Iterable[Tuple[V, V]]:
    return zip(path, path[1:])


## harmonic_backbone_3d

##TODO: 3d SVG using maptplotlib https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html


##TODO:
## velocity plots can also in the harmonic space


##TODO
# can plot density in the cumsum space


##TODO
## can I color edges by edges in the cumsum space?

def harmonic_backbone_3d(adata: AnnData,
                         fate_tree:str =  "fate_tree",
                         backbone_width:float=3,
                         backbone_color:str="black",
                         backbone_joint_size:float=10,
                         backbone_joint_color:str="black",
                         is_color_backbone_joint:bool=False,
                         is_color_backbone:bool=False,
                         #arrow_size:float=0,
                         #arrow_color:str="black",
                         full_traj_matrix:str="full_traj_matrix",
                         clusters:str = "trajs_clusters",
                         evector_name:str = None,
                         retain_clusters:List=[],
                         dims:List[int] = [0,1,2],
                         figsize:Tuple[int,int] = (800,800),
                         ax = None,
                         sample_ratio:float = 0.1,
                         xylabel:bool=True,
                         show_legend = True,
                         fig_path=None,
                         return_fig=False,
                         color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                         **args
                         ):
    """
    plot backbone in cumsum sapce
    1. call plot_trajectory_harmonic_lines
    2. plot fate_tree nodes
    3. plot fate_tree edges
    4. add arrows to the tree edges.


    Parameters
    ----------
    adata: AnnData
        adata object
    fate_tree: str
        fate tree name in adata.uns
    backbone_width: float
        width of the backbone, default 3
    backbone_color: str
        color of the backbone, default black
    backbone_joint_size: float
        size of the backbone joint, default 10
    backbone_joint_color: str
        color of the backbone joint, default black
    is_color_backbone_joint: bool
        if color the backbone joint, default False to color as black
    is_color_backbone: bool
        if color the backbone, default False to color as black
    full_traj_matrix: str
        full trajectory matrix name in adata.uns, default 'full_traj_matrix'
    clusters: str
        clusters name in adata.uns, default 'trajs_clusters'
    evector_name: str
        evector name in adata.uns, default None
    retain_clusters: List
        retain clusters in the clusters, default [] to use all
    dims: List[int]
        dimensions to plot, default [0,1,2]
    figsize: Tuple[int,int]
        figure size, default (800,800)
    ax: matplotlib.axes.Axes
        axes to plot on, default None
    sample_ratio: float
        sample ratio to plot, default 0.1
    xylabel: bool
        if show xy label, default True
    show_legend: bool
        if show legend, default True
    fig_path: str
        figure path to save, default None
    return_fig: bool
        if return the figure, default False
    color_palette: List[str]
        color palette to use, default sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    **args:
        fig.add_scatter3d args
    """
    dims=list(dims)
    if sample_ratio <=0:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Scatter3d(x=[0],
                                     y=[0],
                                     z=[0],
                                     marker=dict(
                                         color='grey',
                                         size=0
                                     ),
                                     showlegend=False))

        fig.update_layout(
                    legend= {'itemsizing': 'constant'}, ## increase the point size in legend.
                    autosize=False,
                    width=figsize[1],
                    height=figsize[0],)

    else:

        fig = plot_trajectory_harmonic_lines_3d(adata,
                                                full_traj_matrix=full_traj_matrix,
                                                clusters = clusters,
                                                evector_name = evector_name,
                                                retain_clusters=retain_clusters,
                                                figsize = figsize,
                                                dims = dims,
                                                show_legend=show_legend,
                                                sample_ratio = sample_ratio,
                                                color_palette = color_palette,
                                                fig_path=None,
                                                return_fig = True,
                                                **args)

    cluster_list = np.array(adata.uns[clusters])
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset


    harmon_x = np.array([adata.uns[fate_tree].nodes[i]['cumsum'][dims[0]] for i in adata.uns[fate_tree].nodes if i !='root'])
    harmon_y = np.array([adata.uns[fate_tree].nodes[i]['cumsum'][dims[1]] for i in adata.uns[fate_tree].nodes if i !='root'])
    harmon_z = np.array([adata.uns[fate_tree].nodes[i]['cumsum'][dims[2]] for i in adata.uns[fate_tree].nodes if i !='root'])

    nodes_name = np.array([i for i in adata.uns[fate_tree].nodes if i !='root'])

    from ..tools.tree_utils import get_tree_leaves_attr
    leaves = get_tree_leaves_attr(adata.uns[fate_tree], attr='original')
    leaf_branches_dict ={} ### trajcluster: branches
    all_leaf_branches = []
    for k,v in leaves.items():
        k_predix = k.split('_')[0]
        branches = [i for i in nodes_name if i.startswith(k_predix)]
        leaf_branches_dict[v] = branches
        all_leaf_branches.extend(branches)

    if is_color_backbone_joint:
        for i, acluster in enumerate(retain_clusters):
            branch_node = leaf_branches_dict[acluster]
            idx = np.where(np.isin(nodes_name, branch_node))[0]
            fig.add_scatter3d(x=harmon_x[idx],
                              y=harmon_y[idx],
                              z=harmon_z[idx],
                              marker=dict(size=backbone_joint_size, color=color_palette[i]),
                              #line=dict(color='green',width=50),
                              showlegend=False,
                              mode='markers',
                             )
        rest_branches = [i for i in nodes_name if i not in all_leaf_branches]
        rest_idx = np.where(np.isin(nodes_name, rest_branches))[0]
        fig.add_scatter3d(x=harmon_x[rest_idx],
                          y=harmon_y[rest_idx],
                          z=harmon_z[rest_idx],
                          marker=dict(size=backbone_joint_size, color=backbone_joint_color),
                          #line=dict(color='green',width=50),
                          showlegend=False,
                          mode='markers',
                         )
    else:
        ## add backbone points
        fig.add_scatter3d(x=harmon_x,
                          y=harmon_y,
                          z=harmon_z,
                          marker=dict(size=backbone_joint_size, color=backbone_joint_color),
                          #line=dict(color='green',width=50),
                          showlegend=False,
                          mode='markers',
                         )


    edges = [e for e in  adata.uns[fate_tree].edges if e[0]!='root']
    edges_end = [e[1] for e in  adata.uns[fate_tree].edges if e[0]!='root']
    pos = {i:adata.uns[fate_tree].nodes[i]['cumsum'][dims[0:3]] for i in adata.uns[fate_tree].nodes if i !='root'}
    lines = [[tuple(pos[e[0]]), tuple(pos[e[1]])] for e in edges]

    if is_color_backbone:
        for i, acluster in enumerate(retain_clusters):
            branch_node = leaf_branches_dict[acluster]
            idxs = np.where(np.isin(edges_end, branch_node))[0]
            for idx in idxs:
                fig.add_scatter3d(x=[lines[idx][0][0], lines[idx][1][0]],
                                  y=[lines[idx][0][1], lines[idx][1][1]],
                                  z=[lines[idx][0][2], lines[idx][1][2]],
                                  mode='lines',
                                  showlegend=False,
                                  line=dict(width=backbone_width, color=color_palette[i]),
                                 )
        rest_idx = np.where(~np.isin(edges_end, all_leaf_branches))[0]
        for idx in rest_idx:
            fig.add_scatter3d(x=[lines[idx][0][0], lines[idx][1][0]],
                              y=[lines[idx][0][1], lines[idx][1][1]],
                              z=[lines[idx][0][2], lines[idx][1][2]],
                              mode='lines',
                              showlegend=False,
                              line=dict(width=backbone_width, color=backbone_color),
                             )

    else:
        ##lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
        for idx in range(len(lines)):
            fig.add_scatter3d(x=[lines[idx][0][0], lines[idx][1][0]],
                              y=[lines[idx][0][1], lines[idx][1][1]],
                              z=[lines[idx][0][2], lines[idx][1][2]],
                              mode='lines',
                              showlegend=False,
                              line=dict(width=backbone_width, color=backbone_color),
                             )
        #lc = mc.LineCollection(lines, colors=backbone_color, linewidths=backbone_width)
        #lc.set_zorder(2)
        #ax.add_collection(lc)


    if not return_fig:
        fig.show()
    if fig_path is not None:
        if fig_path.endswith(".html") or fig_path.endswith(".htm"):
            fig.write_html(fig_path)
        elif fig_path.endswith(".svg") or \
                fig_path.endswith(".pdf") or \
                fig_path.endswith(".eps") or \
                fig_path.endswith(".webp") or \
                fig_path.endswith(".png") or \
                fig_path.endswith(".jpg") or \
                fig_path.endswith(".jpeg"):
            module_check_install("kaleido")
            fig.write_image(fig_path, engine="kaleido")
    return fig if return_fig else None

#endf harmonic_backbone_3d


def harmonic_backbone(adata: AnnData,
                      fate_tree:str =  "fate_tree",
                      backbone_width:float=3,
                      backbone_color:str="black",
                      backbone_joint_size:float=100,
                      backbone_joint_color:str="black",
                      arrow_size:float=0,
                      arrow_color:str="black",
                      full_traj_matrix:str="full_traj_matrix",
                      clusters:str = "trajs_clusters",
                      evector_name:str = None,
                      retain_clusters:List=[],
                      dims:List[int] = [0,1],
                      show_legend:bool=True,
                      legend_loc:str="center left",
                      bbox_to_anchor:List=(1, 0.5),
                      markerscale:float=4,
                      ax = None,
                      sample_ratio:float = 0.1,
                      xylabel:bool=True,
                      color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                      **args
                      ):
    """
    plot backbone in cumsum sapce

    1. call plot_trajectory_harmonic_lines
    2. plot fate_tree nodes
    3. plot fate_tree edges
    4. add arrows to the tree edges.


    Parameters
    ----------
    adata: AnnData
        an Annodata object
    fate_tree: str
        key of the fate tree in adata.uns, default is 'fate_tree'
    backbone_width: float
        width of the backbone lines, default is 3
    backbone_color: str
        color of the backbone lines, default is 'black'
    backbone_joint_size: float
        size of the backbone joints, default is 100
    backbone_joint_color: str
        color of the backbone joints, default is 'black'
    arrow_size: float
        size of the arrows, default is 0
    arrow_color: str
        color of the arrows, default is 'black'
    full_traj_matrix: str
        key of the full trajectory matrix in adata.obsm, default is 'full_traj_matrix'
    clusters: str
        key of the clusters in adata.obs, default is 'trajs_clusters'
    evector_name: str
        key of the eigen vector in adata.uns, default is None
    retain_clusters: List
        list of clusters to be retained, default is [] to use all
    dims: List[int]
        list of dimensions to be used, default is [0,1]
    show_legend: bool
        whether to show legend, default is True
    legend_loc: str
        location of the legend, default is 'center left'
    bbox_to_anchor: List
        bbox_to_anchor of the legend, default is (1, 0.5)
    markerscale: float
        markerscale of the legend, default is 4
    ax: matplotlib axis
        axis to plot the figure, default is None
    sample_ratio: float
        sample ratio of the cells to be plotted, default is 0.1
    xylabel: bool
        whether to show xy labels, default is True
    color_palette: list
        color palette to use, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    """


    from matplotlib import collections  as mc

    def intermediate(p1, p2, nb_points=8):
        """"Return end nb_points equally spaced points
        between p1 and p2"""
        # If we have 8 intermediate points, we have 8+1=9 spaces
        # between p1 and p2
        #print(p1)
        #print(p2)
        x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
        y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

        return (p1[0] + nb_points * x_spacing,  p1[1] +  nb_points * y_spacing)



    dims = [0,1] if len(dims) == 0 else dims
    if len(dims) < 2:
        raise ValueError(f"dims should be at least 2, but {dims} is given!")

    ax = ax or plt.gca()
    plot_trajectory_harmonic_lines(adata,
                                   full_traj_matrix=full_traj_matrix,
                                   clusters=clusters,
                                   evector_name=evector_name,
                                   retain_clusters=retain_clusters,
                                   dims=dims,
                                   show_legend=show_legend,
                                   legend_loc=legend_loc,
                                   bbox_to_anchor=bbox_to_anchor,
                                   markerscale=markerscale,
                                   ax=ax,
                                   sample_ratio=sample_ratio,
                                   xylabel=xylabel,
                                   color_palette=color_palette,
                                   zorder=1,
                                   )
    #print("dims", dims)
    #print("cumsum", adata.uns[fate_tree].nodes["1_18"]['cumsum'])
    #for node in adata.uns[fate_tree].nodes:
    #    print(node, ":")
    #    print(adata.uns[fate_tree].nodes[node]['cumsum'][1])
    harmon_x = [adata.uns[fate_tree].nodes[i]['cumsum'][dims[0]] for i in adata.uns[fate_tree].nodes if i !='root']
    harmon_y = [adata.uns[fate_tree].nodes[i]['cumsum'][dims[1]] for i in adata.uns[fate_tree].nodes if i !='root']
    ax.scatter(harmon_x, harmon_y, s=backbone_joint_size, c=backbone_joint_color, marker='o', alpha=1, edgecolors='none', zorder=3)
    edges = [e for e in  adata.uns[fate_tree].edges if e[0]!='root']

    pos = {i:adata.uns[fate_tree].nodes[i]['cumsum'][dims[0:2]] for i in adata.uns[fate_tree].nodes if i !='root'}
    #lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    lines = [[tuple(pos[e[0]]), tuple(pos[e[1]])] for e in edges]
    lc = mc.LineCollection(lines, colors=backbone_color, linewidths=backbone_width)
    lc.set_zorder(2)
    ax.add_collection(lc)

    if arrow_size > 0:
        headwidth=3 * arrow_size
        headlength=5  * arrow_size
        headaxislength=4.5 * arrow_size
        for i in range(len(lines)):
            x,y = intermediate(p1=lines[i][0], p2=lines[i][1], nb_points=2)
            ax.quiver(x,
                      y,
                      lines[i][1][0]-x,
                      lines[i][1][1]-y,
                      zorder=4,
                      headwidth=headwidth,
                      headlength=headlength,
                      headaxislength=headaxislength,
                      color='black')
#endf harmonic_backbone




def plot_trajectory_harmonic_lines_3d(adata: AnnData,
                                      full_traj_matrix="full_traj_matrix",
                                      clusters = "trajs_clusters",
                                      evector_name = None,
                                      retain_clusters=[],
                                      figsize = (800,800),
                                      dims = [0,1,2],
                                      show_legend=True,
                                      sample_ratio = 0.1,
                                      color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                      fig_path=None,
                                      return_fig = False,
                                      **args):
    """
    Plot the trajectory lines in harmonic ct-map in 3D space

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    full_traj_matrix: str
        key of the full trajectory matrix in adata.uns, default is 'full_traj_matrix'
    clusters: str
        key of the clusters in adata.uns, default is 'trajs_clusters'
    evector_name: str
        key of the eigen vector in adata.uns, default is None
    retain_clusters: list
        clusters to be retained, default is [] to use all
    figsize: tuple
        figure size, default is (800,800)
    dims: list
        dimensions to plot, default is [0,1,2]
    show_legend: bool
        whether to show legend, default is True
    sample_ratio: float
        sample ratio of the cells to be plotted, default is 0.1
    color_palette: list
        color palette to use, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    fig_path: str
        path to save the figure, default is None
    return_fig: bool
        whether to return the figure, default is False
    """

    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if evector_name not in adata.uns:
        raise ValueError(f"{evector_name} not in adata.uns, please check!")
    if full_traj_matrix not in adata.uns:
        raise ValueError(f"{full_traj_matrix} not in adata.uns, please check!")


    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:max(dims)+1],
                                                                     adata.uns[full_traj_matrix])
    return M_plot_trajectory_harmonic_lines_3d(mat_coord_Hspace,
                                        cluster_list = list(adata.uns[clusters]),
                                        retain_clusters=retain_clusters,
                                        dims = dims,
                                        show_legend = show_legend,
                                        sample_ratio = sample_ratio,
                                        color_palette = color_palette,
                                        fig_path=fig_path,
                                        return_fig = return_fig,
                                        **args)


def plot_trajectory_harmonic_lines(adata: AnnData,
                                   full_traj_matrix="full_traj_matrix",
                                   clusters = "trajs_clusters",
                                   evector_name = None,
                                   retain_clusters=[],
                                   dims = [0,1],
                                   show_legend=True,
                                   legend_loc="center left",
                                   bbox_to_anchor=(1, 0.5),
                                   markerscale=4,
                                   ax = None,
                                   sample_ratio = 0.1,
                                   xylabel=True,
                                   color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                   **args):
    """
    Plot the trajectory lines in harmonic ct-map in 2D space

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    full_traj_matrix: str
        key of the full trajectory matrix in adata.uns, default is 'full_traj_matrix'
    clusters: str
        key of the clusters in adata.uns, default is 'trajs_clusters'
    evector_name: str
        key of the eigen vector in adata.uns, default is None
    retain_clusters: list
        clusters to be retained, default is [] to use all
    dims: list
        dimensions to plot, default is [0,1]
    show_legend: bool
        whether to show legend, default is True
    legend_loc: str
        legend location, default is "center left"
    bbox_to_anchor: tuple
        legend bbox_to_anchor, default is (1, 0.5)
    markerscale: float
        legend markerscale, default is 4
    ax: matplotlib axis
        axis to plot on, default is None
    sample_ratio: float
        sample ratio of the cells to be plotted, default is 0.1
    xylabel: bool
        whether to show xy label, default is True
    color_palette: list
        color palette to use, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    """
    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"
    if evector_name not in adata.uns:
        raise ValueError(f"evector_name: {evector_name} not in adata.uns, please check!")
    if clusters not in adata.uns:
        raise ValueError(f"clusters: {clusters} not in adata.uns, please check!")


    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:max(dims)+1],
                                                                   adata.uns[full_traj_matrix])

    M_plot_trajectory_harmonic_lines(mat_coord_Hspace,
                                     cluster_list = list(adata.uns[clusters]),
                                     retain_clusters=retain_clusters,
                                     dims = dims,
                                     show_legend = show_legend,
                                     legend_loc = legend_loc,
                                     bbox_to_anchor = bbox_to_anchor,
                                     markerscale = markerscale,
                                     ax = ax,
                                     sample_ratio = sample_ratio,
                                     xylabel=xylabel,
                                     color_palette = color_palette,
                                     **args)


def plot_trajectory_harmonic_points_3d(adata: AnnData,
                                       full_traj_matrix_flatten="full_traj_matrix_flatten",
                                       clusters = "trajs_clusters",
                                       evector_name = None,
                                       retain_clusters=[],
                                       dims = [0,1,2],
                                       node_size=2,
                                       show_legend=False,
                                       figsize=(800,800),
                                       sample_ratio = 0.1,
                                       color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                       fig_path=None,
                                       return_fig = False,
                                       **args):
    """
    Plot the trajectory groups in harmonic t-map in 3D space

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    full_traj_matrix_flatten: str
        key of the full trajectory matrix in adata.uns, default is 'full_traj_matrix_flatten'
    clusters: str
        key of the clusters in adata.uns, default is 'trajs_clusters'
    evector_name: str
        key of the eigen vector in adata.uns, default is None
    retain_clusters: list
        clusters to be retained, default is [] to use all
    dims: list
        dimensions to plot, default is [0,1,2]
    node_size: float
        node size, default is 2
    show_legend: bool
        whether to show legend, default is False
    figsize: tuple
        figure size, default is (800,800)
    sample_ratio: float
        sample ratio of the cells to be plotted, default is 0.1
    color_palette: list
        color palette to use, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    fig_path: str
        figure path to save, default is None
    return_fig: bool
        whether to return the figure, default is False
    """
    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if evector_name not in adata.uns:
        raise ValueError(f"evector_name: {evector_name} not in adata.uns, please check!")
    if clusters not in adata.uns:
        raise ValueError(f"clusters: {clusters} not in adata.uns, please check!")

    print(max(dims)+1)
    mat_coor_flatten_trajectory = [adata.uns[evector_name][0:max(dims)+1, :] @ mat for mat in adata.uns[full_traj_matrix_flatten].toarray()]
    return M_plot_trajectory_harmonic_points_3d(mat_coor_flatten_trajectory,
                                                cluster_list = list(adata.uns[clusters]),
                                                retain_clusters = retain_clusters,
                                                dims = dims,
                                                node_size = node_size,
                                                show_legend = show_legend,
                                                figsize = figsize,
                                                sample_ratio = sample_ratio,
                                                color_palette = color_palette,
                                                fig_path=fig_path,
                                                return_fig = return_fig,
                                                **args
                                                )


def plot_trajectory_harmonic_points(adata: AnnData,
                                    full_traj_matrix_flatten="full_traj_matrix_flatten",
                                    clusters = "trajs_clusters",
                                    evector_name = None,
                                    retain_clusters=[],
                                    dims = [0,1],
                                    label=True,
                                    labelsize=10,
                                    labelstyle='text',
                                    node_size=2,
                                    show_legend=False,
                                    legend_loc="center left",
                                    bbox_to_anchor=(1, 0.5),
                                    markerscale=4,
                                    ax = None,
                                    sample_ratio = 0.1,
                                    color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                    **args):
    """
    Plot the trajectory groups in harmonic t-map in 2D space


    Parameters
    ---------
    adata: AnnData
        an Annodata object
    full_traj_matrix_flatten: str
        key of the full trajectory matrix in adata.uns, default is 'full_traj_matrix_flatten'
    clusters: str
        key of the clusters in adata.uns, default is 'trajs_clusters'
    evector_name: str
        key of the eigen vector in adata.uns, default is None
    retain_clusters: list
        clusters to be retained, default is [] to use all
    dims: list
        dimensions to plot, default is [0,1]
    label: bool
        whether to show label, default is True
    labelsize: float
        label size, default is 10
    labelstyle: str
        label style, default is 'text'
    node_size: float
        node size, default is 2
    show_legend: bool
        whether to show legend, default is False
    legend_loc: str
        legend location, default is 'center left'
    bbox_to_anchor: tuple
        bbox_to_anchor, default is (1, 0.5)
    markerscale: float
        marker scale, default is 4
    ax: matplotlib axis
        axis to plot, default is None
    sample_ratio: float
        sample ratio of the cells to be plotted, default is 0.1
    color_palette: list
        color palette to use, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    """
    if "graph_basis" in adata.uns.keys() and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if evector_name not in adata.uns:
        raise ValueError(f"evector_name: {evector_name} not in adata.uns, please check!")
    if clusters not in adata.uns:
        raise ValueError(f"clusters: {clusters} not in adata.uns, please check!")


    mat_coor_flatten_trajectory = [adata.uns[evector_name][0:max(dims)+1, :] @ mat for mat in adata.uns[full_traj_matrix_flatten].toarray()]
    M_plot_trajectory_harmonic_points(mat_coor_flatten_trajectory,
                                      cluster_list = list(adata.uns[clusters]),
                                      retain_clusters = retain_clusters,
                                      dims = dims,
                                      label=label,
                                      labelsize=labelsize,
                                      node_size = node_size,
                                      show_legend = show_legend,
                                      legend_loc = legend_loc,
                                      bbox_to_anchor = bbox_to_anchor,
                                      markerscale = markerscale,
                                      ax = ax,
                                      sample_ratio = sample_ratio,
                                      color_palette = color_palette,
                                      **args
                                      )

def plot_fate_tree_embedding(adata: AnnData,
                             graph_name = None,
                             layout_name = None,
                             fate_tree: str = 'fate_tree',
                             bg_node_size=1,
                             bg_node_color='grey',
                             node_size=30,
                             alpha=0.8,
                             label_attr = None,
                             with_labels=False,
                             ax=None,
                             **args,
                             ):
    """
    Plot the fate tree embedding.

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    graph_name: str
        key of the graph in adata.uns, default is None
    layout_name: str
        key of the layout in adata.obsm, default is None
    fate_tree: str
        key of the fate tree in adata.uns, default is 'fate_tree'
    bg_node_size: float
        background node size, default is 1
    bg_node_color: str
        background node color, default is 'grey'
    node_size: float
        node size, default is 30
    alpha: float
        alpha, default is 0.8
    label_attr: str
        label attribute, default is None
    with_labels: bool
        whether to show labels, default is False
    """
    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    ax = plt.gca() if ax is None else ax
    nx.draw_networkx_nodes(adata.uns[graph_name],
                           adata.obsm[layout_name],
                           ax=ax,
                           node_size=bg_node_size,
                           node_color=bg_node_color,
                           alpha=alpha, **args)

    labels = nx.get_node_attributes(adata.uns[fate_tree], label_attr)
    labels = None if len(labels) == 0 else labels
    nx.draw(adata.uns[fate_tree],
            pos=nx.get_node_attributes(adata.uns[fate_tree], layout_name),
            node_size=node_size,
            labels=labels,
            with_labels=with_labels,
            ax=ax)
#endf plot_fate_tree_embedding

def plot_stream_tree_embedding(adata: AnnData,
                             graph_name = None,
                             layout_name = None,
                             stream_tree: str = 'stream_tree',
                             bg_node_size=1,
                             bg_node_color='grey',
                             node_size=30,
                             alpha=0.8,
                             label_attr=None,
                             with_labels=True,
                             ax=None,
                             **args,
                             ):
    """
    Plot stream tree embedding.

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    graph_name: str
        key of the graph in adata.uns, default is None
    layout_name: str
        key of the layout in adata.obsm, default is None
    stream_tree: str
        key of the stream tree in adata.uns, default is 'stream_tree'
    bg_node_size: float
        background node size, default is 1
    bg_node_color: str
        background node color, default is 'grey'
    node_size: float
        node size, default is 30
    alpha: float
        alpha, default is 0.8
    label_attr: str
        label attribute, default is None
    with_labels: bool
        whether to show labels, default is True
    ax: matplotlib axis
        axis to plot, default is None
    """

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]


    ax = plt.gca() if ax is None else ax
    nx.draw_networkx_nodes(adata.uns[graph_name],
                           adata.obsm[layout_name],
                           ax=ax,
                           node_size=bg_node_size,
                           node_color=bg_node_color,
                           alpha=alpha, **args)

    labels = nx.get_node_attributes(adata.uns[stream_tree], label_attr)
    labels = None if len(labels) == 0 else labels
    nx.draw(adata.uns[stream_tree],
            pos=nx.get_node_attributes(adata.uns[stream_tree], layout_name),
            node_size=node_size,
            labels=labels,
            with_labels=with_labels,
            ax=ax)
#endf plot_fate_tree_embedding



def plot_fate_tree(adata: AnnData,
                   fate_tree: str = 'fate_tree',
                   layout_prog = 'twopi',
                   with_labels= True,
                   ax=None,
                   **args
                   ):
    """
    re-calcuate layout of a tree and plot

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    fate_tree: str
        key of the fate tree in adata.uns, default is 'fate_tree'
    layout_prog: str
        layout algorithm, can be: dot, twopi, fdp, sfdp, circo
    with_labels: bool
        whether to show labels, default is True
    ax: matplotlib axis
        axis to plot, default is None
    """
    ax = plt.gca() if ax is None else ax
    pos =nx.nx_pydot.graphviz_layout(adata.uns[fate_tree].to_undirected(), prog=layout_prog)
    nx.draw(adata.uns[fate_tree], pos, with_labels=with_labels, ax=ax, **args)


def plot_density_grid(adata: AnnData,
                      graph_name = None,
                      layout_name = None,
                      cluster_name = "trajs_clusters",
                      trajs_name = "knn_trajs",
                      retain_clusters=[],
                      sample_n=10000,
                      figsize=(20,16),
                      title_prefix='cluster_',
                      bg_alpha = 0.5,
                      node_size = 2,
                      return_fig = False,
                      **args
                      ):
    """
    plot density grid of clusters


    Parameters
    ---------
    adata: AnnData
        an Annodata object
    graph_name: str
        key of the graph in adata.uns, default is None
    layout_name: str
        key of the layout in adata.obsm, default is None
    cluster_name: str
        key of the cluster in adata.obs, default is 'trajs_clusters'
    trajs_name: str
        key of the trajs in adata.uns, default is 'knn_trajs'
    retain_clusters: list
        list of clusters to plot, default is [] to use all
    sample_n: int
        number of cells to sample, default is 10000
    figsize: tuple
        figure size, default is (20,16)
    title_prefix: str
        prefix of the title, default is 'cluster_'
    bg_alpha: float
        background alpha, default is 0.5
    node_size: float
        node size, default is 2
    return_fig: bool
        whether to return the figure, default is False
    """

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]




    if graph_name not in adata.uns.keys():
        raise ValueError("graph_name not in adata.uns.keys()")
    if layout_name not in adata.obsm.keys():
        raise ValueError("layout_name not in adata.obsm.keys()")
    if cluster_name not in adata.uns.keys():
        raise ValueError("cluster_name not in adata.uns.keys()")
    if trajs_name not in adata.uns.keys():
        raise ValueError("trajs_name not in adata.uns.keys()")

    fig, axes = G_plot_density_grid(adata.uns[graph_name],
                        adata.obsm[layout_name],
                        adata.uns[cluster_name],
                        adata.uns[trajs_name],
                        retain_clusters=retain_clusters,
                        sample_n=sample_n,
                        figsize=figsize,
                        title_prefix=title_prefix,
                        bg_alpha=bg_alpha,
                        node_size=node_size,
                        **args)
    return None if not return_fig else (fig, axes)


def plot_trajs_embedding(adata,
                         embedding = "trajs_dm",
                         clusters = "trajs_clusters",
                         node_size=1,
                         label=True,
                         labelsize=15,
                         labelstyle='text',
                         show_legend=True,
                         markerscale=10,
                         ax = None,
                         **args
                         ):
    """
    Plot trjactories edges embeddings in umap or original t-map space

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    embedding: str
        key of the embedding in adata.uns, indicate edges embedding, default is 'trajs_dm'
    clusters: str
        key of the clusters in adata.uns, default is 'trajs_clusters'
    node_size: float
        node size, default is 1
    label: bool
        whether to show labels, default is True
    labelsize: float
        label size, default is 15
    labelstyle: str
        label style, default is 'text'
    show_legend: bool
        whether to show legend, default is True
    markerscale: float
        marker scale, default is 10
    ax: matplotlib axis
        axis to plot, default is None
    """
    ax = plt.gca() if ax is None else ax
    if embedding not in adata.uns.keys():
        raise ValueError("embedding not in adata.uns.keys()")
    if clusters not in adata.uns.keys():
        raise ValueError("clusters not in adata.uns.keys()")

    plot_embedding(adata.uns[clusters],
                   adata.uns[embedding],
                   node_size=node_size,
                   label=label,
                   labelsize=labelsize,
                   labelstyle=labelstyle,
                   show_legend=show_legend,
                   markerscale=markerscale,
                   ax = ax,
                   **args)


def plot_eigen_line(adata: AnnData,
                    evalue_name=None,
                    n_eig=10,
                    step_size=1,
                    show_legend=True,
                    ax=None,
                    **args):
    """
    plot eigen values line to check the elbow of them

    parameters
    --------
    adata: AnnData
        an Annodata object
    evalue_name: str
        eigen values of stored in adata.uns get from graph_basis if None
    n_eig: int
        number of eigen values to plot
    step_size: int
        ticks for the plotting
    show_legend: bool
        whether to show legend, default is True
    ax: matplotlib axis
        axis to plot, default is None
    """
    if "graph_basis" in adata.uns and not evalue_name:
        evalue_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_value"
    if evalue_name not in adata.uns:
        raise ValueError(f"{evalue_name} is not there, please check adata.uns")
    if n_eig <=0:
        print("number of eigen values should be positive, reset to 10")
        n_eig = 10
    if step_size <=0:
        print("step size should be positive, reset to 1")
        step_size = 1
    L_plot_eigen_line(adata.uns[evalue_name],
                      n_eig=n_eig,
                      step_size=step_size,
                      show_legend=show_legend,
                      ax=ax,
                      **args)
#endf

def plot_traj(adata: AnnData,
              graph_name: str = None,
              layout_name: str=None,
              holes: List[List[int]] = None,
              trajectory: Union[List[int], np.ndarray] = None,
              colorid=None,
              hole_centers=None,
              *,
              ax: Optional[plt.Axes] = None,
              node_size=5,
              edge_width=1,
              plot_node=True,
              alpha_nodes = 0.3,
              color_palette = sns.color_palette('tab10'),
              **args
              ):
    """
    plot the trajectory on the graph

    Parameters
    ---------
    adata: AnnData
        an Annodata object
    graph_name: str
        key of the graph in adata.uns, default is None
    layout_name: str
        key of the layout in adata.obsm, default is None
    holes: List[List[int]]
        holes in the graph, default is None
    trajectory: Union[List[int], np.ndarray]
        trajectory to plot, default is None
    colorid: List[int]
        color id for each node, default is None
    hole_centers: List[int]
        hole centers, default is None
    ax: matplotlib axis
        axis to plot, default is None
    node_size: float
        node size, default is 5
    edge_width: float
        edge width, default is 1
    plot_node: bool
        whether to plot nodes, default is True
    alpha_nodes: float
        alpha for nodes, default is 0.3
    color_palette: List[str]
        color palette, default is sns.color_palette('tab10')
    """
    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation"

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if trajectory is None:
        raise ValueError("trajectory is None")

    G_plot_traj(adata.uns[graph_name],
                adata.obsm[layout_name],
                trajectory=trajectory,
                colorid=colorid,
                holes=holes,
                hole_centers=hole_centers,
                ax=ax,
                node_size=node_size,
                edge_width=edge_width,
                plot_node=plot_node,
                alpha_nodes=alpha_nodes,
                color_palette=color_palette,
                **args)



def nxdraw_harmonic(adata: AnnData,
                    graph_name: str=None,
                    evector_name: str=None,
                    title: str= "",
                    dims:List[V] = [0,1],
                    node_size:float=1,
                    show_center = True,
                    with_potential = 'u',
                    edge_cell_types = None,
                    show_legend = True,
                    markerscale = 1,
                    ax = None,
                    color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                    seed =2022,
                    **args):
    """
    plot holes from eigen decomposition of L1

    Parameters
    -------
    adata: AnnData
        an Annodata object
    graph_name: str
        key of the graph in adata.uns, default is None
    evector_name: str
        eigen vector of L1
    title: str
        title of the plot
    dims: list
        the dimension of the eigen vector to plot
    node_size: int
        the size of the node
    show_center: bool
        whether to show the center of the hole
    with_potential: str
        use potential to flip the evector or not
    edge_cell_types: list
        cell types to plot
    show_legend: bool
        whether to show legend
    markerscale: float
        scale of the marker
    ax: matplotlib axis
        the axis to plot
    color_palette: list
        color palette to  use default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    seed: int
        random seed
    """
    ## pie show the cell types, would be really slow
    ## randomly select an end to specify the celltype.
    from ..tools.tree_utils import _edge_two_ends

    ax = ax or plt.gca()

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"
    if "graph_basis" in adata.uns and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"


    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns, please check!")
    if evector_name not in adata.uns:
        raise ValueError(f"{evector_name} not in adata.uns, please check!")

    H0 = adata.uns[evector_name][dims[0]]
    H1 = adata.uns[evector_name][dims[1]]

    if with_potential is not None:
        elist = adata.uns[graph_name].edges()
        u=nx.get_node_attributes(adata.uns[graph_name], with_potential)
        direction = [-1 if (u[e]-u[a])>0 else 1 for a,e in elist]
        H0 = [i*j for i,j in zip(H0, direction)]
        H1 = [i*j for i,j in zip(H1, direction)]

    if edge_cell_types is not None:
        edge_ends_dict = _edge_two_ends(adata, graph_name=graph_name)
        if edge_cell_types not in adata.obs.keys():
            raise ValueError("edge_cell_types not in adata.obs")
        celltypes = adata.obs[edge_cell_types]

        # random select an end of an edge.
        np.random.seed(seed)
        rand_ct_idx = np.random.randint(2, size=len(H0))
        cts = [celltypes[edge_ends_dict[i][rand_ct_idx[i]]] for i in range(len(H0))]
        for i, ct in enumerate(np.unique(cts)):
            idx = np.where(np.array(cts)==ct)[0]
            ax.scatter(np.array(H0)[idx], np.array(H1)[idx], s=node_size, label=ct, c=color_palette[i], **args)
        if show_legend:
            ax.legend(markerscale=markerscale, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.scatter(H0, H1, s=node_size,  **args)
    if show_center:
        ax.scatter(0, 0, s=node_size*5,  c='black')

    if title:
        ax.set_title(title)

#endf nxdraw_Holes





def nxdraw_holes(adata: AnnData,
                 graph_name: str=None,
                 layout_name: str=None,
                 evector_name: str=None,
                 title: str= "",
                 edge_value: List[V] = [],
                 vector_dim:int=0,
                 font_size:int=0,
                 node_size:float=1,
                 width:int=1,
                 edge_cmap=plt.cm.RdBu_r,
                 with_labels: bool = False,
                 with_potential = 'u',
                 flip: bool = False,
                 is_norm: bool=False,
                 is_abs: bool = False,
                 is_arrow: bool = True,
                 ax = None,
                 colorbar=False,
                 **args):
    """
    plot holes from eigen decomposition of L1

    Parameters
    ----
    adata: AnnData
        an Annodata object
    graph_name: str
        graph_name of the starts ends connected graph
    layout_name: str
        layout name show the graph
    evector_name: str
        eigen vector of L1
    title: str
        title of the plot
    edge_value: list
        the value of edges, if not None, use edge_value to plot the edge
    vector_dim: int
        the dimension of the eigen vector to plot
    font_size: int
        the size of the font
    node_size: int
        the size of the node
    width: int
        the width of the edge
    edge_cmap: matplotlib colormap
        the colormap of the edge
    with_labels: bool
        show the label of the node if True
    with_potential: None or length of 2 list,
        use potential u to calculate the direction of an edge,
                    eigen values * -1 if uend - ustart > 0 else eigen values * 1
    flip: bool
        flip the direction of an edge
    is_arrow:bool
        show directed graph if True else undirected
    is_norm: bool
        normalize the eigen value to [0,1]
    is_abs: bool
        take absolute value of eigen value
    ax: matplotlib axis
        the axis to plot
    colorbar: bool
        show colorbar or not
    """

    ax = ax or plt.gca()


    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"
    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"] + "_triangulation_circle"
    if "graph_basis" in adata.uns and not evector_name:
        evector_name = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_vector"

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns, please check!")
    if evector_name not in adata.uns:
        raise ValueError(f"{evector_name} not in adata.uns, please check!")

    H = adata.uns[evector_name][vector_dim]
    if is_norm:
        H = np.abs(norm01(H)) if is_abs else norm01(H)
    if len(edge_value)>0:
        H = np.abs(edge_value) if is_abs else edge_value

    elist = adata.uns[graph_name].edges()
    #elist_set = set(list(elist))
    if with_potential is not None:
        u=nx.get_node_attributes(adata.uns[graph_name], with_potential)
        direction = [-1 if (u[e]-u[a])>0 else 1 for a,e in elist]
        H = [i*j for i,j in zip(H, direction)]

    #gg = adata.uns[graph_name].to_directed()
    #for edge in list(gg.edges()):
    #    if edge in elist_set:
    #        continue
    #    gg.remove_edge(edge[0], edge[1])

    if flip:
        H = [i*-1 for i in H]
    absH = np.abs(H)
    width_range = [i*width for  i in (absH - min(absH))/(max(absH) - min(absH))]
    nx.draw_networkx(adata.uns[graph_name] if is_arrow else adata.uns[graph_name].to_undirected(),
                     adata.obsm[layout_name],
                     edge_color=list(H),
                     node_size=node_size,
                     width=width_range,
                     edge_cmap=edge_cmap,
                     with_labels=with_labels,
                     ax=ax,
                     **args)

    if title:
        ax.set_title(title)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin = min(H), vmax=max(H)))
        sm._A = []
        plt.colorbar(sm, ax=plt.gca())
#endf nxdraw_Holes


def nxdraw_score(adata: AnnData,
                 graph_name:str = None,
                 layout_name:str = None,
                 color:str = "u",
                 colorbar:bool = True,
                 directed:bool = False,
                 font_size:float = 0,
                 cmap = plt.cm.get_cmap('viridis'),
                 **args
    ):
    """
    networkx plot continous values

    Parameters
    ----------
    adata: AnnData
        an Annodata object
    graph_name: str
        graph_name of the starts ends connected graph
    layout_name: str
        layout name show the graph
    color: str
        the color of the node
    colorbar: bool
        show colorbar or not
    directed: bool
        show directed graph or not
    font_size: float
        the size of the font
    cmap: matplotlib colormap
        the colormap of the node
    """

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]


    u_color = None
    if color in set(chain.from_iterable(d.keys() for *_, d in adata.uns[graph_name].nodes(data=True))):
        u_color = np.fromiter(nx.get_node_attributes(adata.uns[graph_name], color).values(), dtype='float')
    elif color in adata.obs:
        u_color = adata.obs[color].values

    ##TODO assert digital

    if directed:
        nx.draw_networkx(adata.uns[graph_name], adata.obsm[layout_name], node_color=u_color, cmap=cmap, font_size=font_size, with_labels=False, **args)
    else:
        nx.draw_networkx(adata.uns[graph_name].to_undirected(), adata.obsm[layout_name], cmap=cmap, node_color=u_color, font_size=font_size, with_labels=False, **args)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = min(u_color), vmax=max(u_color)))
        sm._A = []
        plt.colorbar(sm, ax=plt.gca())
#endf nxdraw_score



def nxdraw_group(adata: AnnData,
                 graph_name:str = None,
                 layout_name: str = None,
                 group_name:str = 'group',
                 show_edges:bool=True,
                 show_legend:bool=True,
                 color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                 legend_loc="center left",
                 bbox_to_anchor=(1, 0.5),
                 markerscale=1,
                 label=True,
                 labelsize=10,
                 labelstyle='text',
                 directed =False,
                 edge_color='gray',
                 legend_col=1,
                 ax = None,
                 **args):
    """
    networkx plot groups

    Parameters
    ----------
    adata: AnnData
        an Annodata object
    graph_name: str
        graph_name of the starts ends connected graph
    layout_name: str
        layout name show the graph
    group_name: str
        the group name of the node
    show_edges: bool
        show edges or not
    show_legend: bool
        show legend or not
    color_palette: list
        the color palette of the group, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    legend_loc: str
        the location of the legend
    bbox_to_anchor: tuple
        the bbox_to_anchor of the legend
    markerscale: float
        the scale of the marker
    label: bool
        show label or not
    labelsize: float
        the size of the label
    labelstyle: str
        the style of the label
    directed: bool
        show directed graph or not
    edge_color: str
        the color of the edge
    legend_col: int
        the number of the legend column
    ax: matplotlib.axes
        the axes of the plot
    """
    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]


    G_nxdraw_group(adata.uns[graph_name],
                   adata.obsm[layout_name],
                   adata.obs[group_name],
                   show_edges=show_edges,
                   show_legend=show_legend,
                   color_palette=color_palette,
                   legend_loc=legend_loc,
                   bbox_to_anchor=bbox_to_anchor,
                   markerscale=markerscale,
                   label=label,
                   edge_color=edge_color,
                   labelsize=labelsize,
                   labelstyle=labelstyle,
                   directed = directed,
                   legend_col = legend_col,
                   ax=ax,
                   **args)


#endf nxdraw_group

def plot_triangle_density(adata: AnnData,
                          graph_name: str = None,
                          layout_name: str = None,
                          node_size=10,
                          ax=None,
                          cmap = plt.get_cmap("jet"),
                          colorbar = True,
                          **args):
    """
    plot triangle density values of the trianglulated graph

    Parameters
    ----------
    adata: AnnData
        an Annodata object
    graph_name: str
        graph_name of the starts ends connected graph
    layout_name: str
        layout name show the graph
    node_size: int
        the size of the node
    ax: matplotlib.axes
        the axes of the plot
    cmap: matplotlib colormap
        the colormap of the node
    colorbar: bool
        show colorbar or not
    """

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"] + "_triangulation_circle"


    G_plot_triangle_density(adata.uns[graph_name], adata.obsm[layout_name], node_size=node_size, ax=ax, cmap=cmap, colorbar=colorbar, **args)
#endf plot_triangle_density



def plot_pie_fate_tree(adata: AnnData,
                       graph_name=None,
                       layout_name=None,
                       fate_tree='stream_tree',
                       piesize=0.05,
                       group= 'group',
                       show_nodes=True,
                       show_legend = False,
                       legend_column = 3,
                       color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                       bg_alpha=0.2,

                       ax = None,
                       ):
    """
    Piechart show cell type proportions along Trajectory tree in 2d space

    Parameters
    ----------
    adata: AnnData
        an Annodata object
    graph_name: str
        graph_name of the starts ends connected graph
    layout_name: str
        layout name show the graph
    fate_tree: str
        the fate tree name
    piesize: float
        the size of the pie
    group: str
        the group name of the node
    show_nodes: bool
        show nodes or not
    show_legend: bool
        show legend or not
    legend_column: int
        the number of the legend column
    color_palette: list
        the color palette of the group, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    bg_alpha: float
        the alpha of the background
    ax: matplotlib.axes
        the axes of the plot
    """
    #https://www.appsloveworld.com/coding/python3x/146/creating-piechart-as-nodes-in-networkx

    ax = ax or plt.gca()
    if fate_tree not in adata.uns.keys():
        raise ValueError("fate_tree not found in adata.uns")

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns.keys() and not layout_name:
        layout_name= adata.uns["graph_basis"]


    if show_nodes:
        nx.draw_networkx_nodes(adata.uns[graph_name],
                               pos=adata.obsm[layout_name],
                               ax=ax,
                               node_size=3,
                               node_color='grey',
                               alpha=bg_alpha)

    node_pos = nx.get_node_attributes(adata.uns[fate_tree],layout_name)
    nx.draw_networkx_edges(adata.uns[fate_tree], pos=node_pos, ax=ax)

    dic = get_nodes_celltype_counts(adata, tree_name='fate_tree', edge_attr='ecount', cluster=group)


    all_nodes = list(set(adata.obs[group]))
    bbox = ax.get_position().get_points()
    ax_x_min = bbox[0, 0]
    ax_x_max = bbox[1, 0]
    ax_y_min = bbox[0, 1]
    ax_y_max = bbox[1, 1]
    xlim = ax_x_min, ax_x_max
    ylim = ax_y_min, ax_y_max


    trans=ax.transData.transform
    trans2 = ax.transAxes.inverted().transform

    p2=piesize/2.0
    pie_axs = []
    for node in adata.uns[fate_tree].nodes():
        attributes =[np.sqrt(dic[node].get(c, 0)+0) for c in all_nodes]
        xx,yy=trans(node_pos[node]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        xa = xlim[0] + (xa - piesize / 2) * (xlim[1]-xlim[0])
        ya = ylim[0] + (ya - piesize / 2) * (ylim[1]-ylim[0])
        if ya < 0: ya = 0
        if xa < 0: xa = 0
        rect = [xa, ya, piesize * (xlim[1]-xlim[0]), piesize * (ylim[1]-ylim[0])]

        a = plt.axes(rect, frameon=False)
        # for legend
        pie_axs.append(a)
        a.set_aspect('equal')
        a.pie(attributes, labels=all_nodes, labeldistance=None, colors=color_palette)


    if show_legend:
        a.legend(all_nodes, loc='center left', bbox_to_anchor=(1, 0.5), ncol=legend_column)



def G_nxdraw_group(g,
                 layouts,
                 groups,
                 show_edges:bool=True,
                 show_legend:bool=True,
                 color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                 legend_loc="center left",
                 bbox_to_anchor=(1, 0.5),
                 markerscale=1,
                 label=True,
                 labelsize=10,
                 edge_color='gray',
                 labelstyle='text',
                 directed=False,
                 legend_col=1,
                 ax = None,
                 **args):
    """
    Parameters
    ---------
    g: networkx graph
        graph to plot
    layouts: layouts dict or array
        layouts to plot
    groups: list
        edges clusters
    show_edges: bool
        if show edges
    color_palette: list
        color palette for show groups
    legend_loc: str
        legend location
    bbox_to_anchor: tuple
        for position of the legend
    markerscale: float
        legend marker scale to larger or smaller
    label: bool
        if show label
    labelsize: int
        labelsize
    labelstyle: str
        options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    directed: bool
        if the graph is directed
    legend_col: int
        the number of the legend column
    ax: matplotlib.axes
        the axes of the plot
    """
    ax = ax or plt.gca()
    mapping = dict(zip(sorted(groups),itertools.count()))
    rev_mapping = {v:k for k,v in mapping.items()}
    colors = [mapping[groups[n]] for n in range(len(groups))]
    d_colors = defaultdict(list)
    d_group = {}
    for v, k in enumerate(colors):
        d_colors[k] = d_colors[k] + [v]
        d_group[k] = groups[v]

    if label:
        labeldf = pd.DataFrame(layouts).T if isinstance(layouts, dict) else pd.DataFrame(layouts)
        labeldf.columns = ['x', 'y']
        labeldf['label'] = list(groups)
        #print(labeldf)

    if show_edges:
        if directed:
            nx.draw_networkx_edges(g, pos=layouts, ax=ax, edge_color=edge_color)
        else:
            nx.draw_networkx_edges(g.to_undirected(), pos=layouts, ax=ax, edge_color=edge_color)

    for i, (k, v) in enumerate(d_colors.items()):
        name = k
        nodes = v
        nx.draw_networkx_nodes(g, pos=layouts, nodelist=nodes, ax=ax,
                               node_color=color_palette[i], label=rev_mapping[name], **args)
        if label:
            if labelstyle=='text' or labelstyle == "color":
                ax.annotate(d_group[k],
                        labeldf.loc[labeldf['label']==d_group[name],['x','y']].median(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="black" if labelstyle == "text" else color_palette[i])
            elif labelstyle == "box":
                ax.annotate(d_group[k],
                        labeldf.loc[labeldf['label']==d_group[name],['x','y']].median(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="white",
                        backgroundcolor=color_palette[i])
            else:
                print("warning, labelstyle is not correct, options: color, text, box")


    if show_legend:
            ax.legend(loc=legend_loc,  bbox_to_anchor=bbox_to_anchor, markerscale=markerscale, ncol=legend_col)



def G_plot_traj(graph: nx.Graph,
                node_positions: np.ndarray,
                holes: List[List[int]] = None,
                trajectory: List = None,
                colorid=None,
                hole_centers=None,
                *,
                ax: Optional[plt.Axes] = None,
                node_size=5,
                edge_width=1,
                plot_node=True,
                alpha_nodes = 0.3,
                color_palette = sns.color_palette('tab10'),
                **args,
                ) -> None:

    """
    Parameters
    ----------
    graph: nx.Graph
        graph to plot
    node_positions: str
        layout array
    holes: List[List[int]]
        holes in the graph, default is None
    trajectory: Union[List[int], np.ndarray]
        trajectory to plot, default is None
    colorid: List[int]
        color id for each node, default is None
    hole_centers: List[int]
        hole centers, default is None
    ax: matplotlib axis
        axis to plot, default is None
    node_size: float
        node size, default is 5
    edge_width: float
        edge width, default is 1
    plot_node: bool
        whether to plot nodes, default is True
    alpha_nodes: float
        alpha for nodes, default is 0.3
    color_palette: List[str]
        color palette, default is sns.color_palette('tab10')
    """
    ax = ax or plt.gca()
    try:
        if holes is not None:
            patches = []
            for hole in holes:
                corners = sort_corners_of_hole(graph, hole)
                corners = np.array(
                    list(map(lambda point: node_positions[point], corners)))
                patches.append(Polygon(corners, facecolor='#9c9e9f', closed=True))

            ax.add_collection(PatchCollection(
                patches, alpha=0.7, match_original=True))
    except RuntimeError:
        print('Error with graph')

    if plot_node:
        nx.draw_networkx_nodes(graph, node_positions,
                               ax=ax, node_size=node_size,
                               node_color='#646567',
                               alpha= alpha_nodes
                               )

    if trajectory:
        if not colorid:
            colorid = 0
        color = color_palette[colorid]
        # We use a bigger node size here so that the arrows do not
        # fully reach the nodes. This makes the visualization a bit
        # better.


        if all(isinstance(elem, list) for elem in trajectory):
            for a_traj in trajectory:
                nx.draw_networkx_edges(graph,
                                       node_positions,
                                       ax=ax,
                                       edgelist=list(edges_on_path(a_traj)),
                                       node_size=10,
                                       width=edge_width,
                                       edge_color=color,
                                       arrows=True,
                                       arrowstyle='->',
                                       **args,
                                       )
        else:
            nx.draw_networkx_edges(graph,
                                   node_positions,
                                   ax=ax,
                                   edgelist=list(edges_on_path(trajectory)),
                                   node_size=10,
                                   width=edge_width,
                                   edge_color=color,
                                   arrows=True,
                                   arrowstyle='->',
                                   **args,
                                   )

    if hole_centers is not None:
        ax.scatter(x=hole_centers[:, 0], y=hole_centers[:, 1], marker='x')



def G_plot_triangle_density(g:nx.Graph,
                          layouts,
                          node_size=10,
                          ax=None,
                          cmap = plt.get_cmap("jet"),
                          colorbar = True,
                          **args):

    """
    Parameters
    ---------
    g: nx.Graph
        networkx graph
    layouts: layouts dict or array
        layout of the graph
    ax: matplotlib axis
        axis to plot, default is None
    cmap: matplotlib.colors.LinearSegmentedColormap
        colormap, default is plt.get_cmap("jet")
    colorbar: bool
        if show colorbar
    """
    ax = ax or plt.gca()
    values = nx.triangles(g.to_undirected())
    n_color = np.asarray([values[n] for n in g.nodes()])
    nx.draw(g, layouts, node_color=n_color, node_size=node_size, ax=ax, cmap=cmap, **args)
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(n_color), vmax=max(n_color)))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca())
#endf plot_triangle_density



def plot_embedding(cluster_list = [],
                   embedding = None,
                   color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                   retain_clusters=[],
                   node_size= 4,
                   label=True,
                   labelsize=10,
                   labelstyle='text',
                   ax=None,
                   show_legend=True,
                   legend_loc = "center left",
                   bbox_to_anchor = (1,0.5),
                   markerscale =5,
                   facecolor='white',
                   **args
                   ):

    """
    Parameters
    ---------
    cluster_list: list
        cluster labels for each point
    embedding: np.ndarray
        embeddings, shape should be nx2
    color_palette: list
        color palette, default is sns.color_palette(cc.glasbey, n_colors=50).as_hex()
    retain_clusters: list
        which clusters to plot
    node_size: int
        node size
    label: bool
        if add labels
    labelsize: int
        size of labels
    labelstyle: str
        options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    ax: matplotlib axis
        matplotlib ax
    show_legend: bool
        if show_legend
    legend_loc: str
        legend location
    bbox_to_anchor: tuple
        tune of legend position
    markerscale: int
        legend markerscale
    facecolor: str
        plt background
    **args:
        parameters of ax.scatter
    """
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    if len(cluster_list)==0 or embedding is None:
        print("Error: cluster_list and embedding should be not None!")
        return
    #print(len(cluster_list), embedding.shape[0])
    assert(len(cluster_list) == embedding.shape[0])
    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset

    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)

    ax = ax or plt.gca()

    if label:
       labeldf = pd.DataFrame(embedding)
       labeldf.columns = ['x', 'y']
       labeldf['label'] = list(cluster_list)

    cluster_n = len(set(cluster_list))
    ax.set_facecolor(facecolor)
    for i, x in enumerate(retain_clusters):
        idx = [i for i in np.where(cluster_list == x)[0]]
        ax.scatter(x=embedding[idx, 0], y=embedding[idx, 1], c = color_palette[i], s=node_size, **args, label=x)
        if label:
            if labelstyle=='text' or labelstyle == "color":
                ax.annotate(x,
                        labeldf.loc[labeldf['label']==x,['x','y']].median(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="black" if labelstyle == "text" else color_palette[i])
            elif labelstyle == "box":
                ax.annotate(x,
                        labeldf.loc[labeldf['label']==x,['x','y']].median(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="white",
                        backgroundcolor=color_palette[i])
            else:
                print("warning, labelstyle is not correct, options: color, text, box")

    if show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, markerscale=markerscale)



def G_plot_density_grid(G,
                      layouts,
                      cluster_list,
                      traj_list,
                      retain_clusters=[],
                      sample_n=10000,
                      figsize=(20,16),
                      title_prefix='cluster_',
                      sharex = True,
                      sharey = True,
                      bg_alpha = 0.5,
                      node_size = 2,
                      colorbar=False,
                      **args
                      ):

    """
    Parameters
    ---------
    G: nx.Graph
        networkx graph
    layouts:
        layouts dict or array
    cluster_list: list
        cluster label list
    retain_clusters: list
        only show clusters in retain_clusters
    sample_n: int
        sample the for the kde estimate to save time
    figsize: tuple
        figure size tuple
    title_prefix: str
        prefix for each subplot
    sharex: bool
        if share x axis
    sharey: bool
        if share y axis
    bg_alpha: float
        background alpha
    node_size: int
        density node size
    colorbar: bool
        if show colorbar
    """
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)

    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset
    cluster_n = len(retain_clusters)
    assert(cluster_n > 0)

    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)

    if not isinstance(traj_list, np.ndarray):
        traj_list = np.array(traj_list, dtype=object)

    r,c = get_uniform_multiplication(cluster_n)
    r = r - 1 if (r-1)*c == cluster_n else r
    fig, axes = plt.subplots(r,c, sharex=sharex, sharey=sharey,)
    fig.set_size_inches(figsize[0], figsize[1])
    z_min = 1e100
    z_max = -1e100
    for a, cluster in enumerate(retain_clusters):
        i = int(a/c)
        j = a%c
        idx = [i for i in np.where(cluster_list == cluster)[0]]
        cdic = kde_eastimate(traj_list[idx], layouts, sample_n)
        x = cdic['x']
        y = cdic['y']
        z = cdic['z']
        z_min = min(z_min, np.min(z))
        z_max = max(z_max, np.max(z))
        if r == 1 and c == 1:
            nx.draw_networkx_nodes(G, layouts, ax = axes, node_size=node_size, node_color='grey', alpha=bg_alpha)
            axes.scatter(x, y, c=z, s=node_size, **args)
            axes.set_title(f"{title_prefix}{cluster}")
        elif r == 1:
            nx.draw_networkx_nodes(G, layouts, ax = axes[j], node_size=node_size, node_color='grey', alpha=bg_alpha)
            axes[j].scatter(x, y, c=z, s=node_size, **args)
            axes[j].set_title(f"{title_prefix}{cluster}")
        else:
            nx.draw_networkx_nodes(G, layouts, ax = axes[i,j], node_size=node_size, node_color='grey', alpha=bg_alpha)
            axes[i,j].scatter(x, y, c=z, s=node_size, **args)
            axes[i,j].set_title(f"{title_prefix}{cluster}")

    ## axis off
    for a in range(cluster_n, r*c):
        i = int(a/c)
        j = a%c
        if r == 1 and c == 1:
            axes.axis('off')
        elif r == 1:
            axes[j].axis('off')
        else:
            axes[i,j].axis('off')

    print("z_min, z_max", z_min, z_max)
    if colorbar:
        cmap = plt.get_cmap("viridis")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = z_min, vmax=z_max))
        sm._A = []
        plt.colorbar(sm, ax=plt.gca())

    return fig, axes



def M_plot_trajectory_harmonic_lines_3d(mat_coord_Hspace,
                                        cluster_list,
                                        retain_clusters=[],
                                        dims = [0,1,2],
                                        figsize = (800, 800),
                                        show_legend=True,
                                        sample_ratio = 1,
                                        color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                        fig_path = None,
                                        return_fig=False,
                                        **args):
    """
    Parameters
    ---------
    mat_coord_Hspace:
        matrix of coordinates in the Hspace
    cluster_list:
        cluster_list for each trajectory
    retain_clusters:
        only show clusters in retain_clusters
    dims:
        dimensions to plot
    figsize:
        figure size
    show_legend:
        if show legend
    color_palette:
        color palette for show cluster_list
    fig_path:
        figure path
    return_fig:
        if return fig
    """
    import plotly.graph_objects as go

    assert(all(np.array(dims) < mat_coord_Hspace[0].shape[0])) ## dims is in the range of the dimension of the data
    assert(len(dims) >=3)
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    #print(retain_clusters)
    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset

    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)


    cumsums = list(map(lambda i: [np.cumsum(i[dims[0]]), np.cumsum(i[dims[1]]), np.cumsum(i[dims[2]])], mat_coord_Hspace))
    is_first = True
    for i, cluster in enumerate(retain_clusters):
        #print(i, cluster)
        v = [i for i in np.where(cluster_list == cluster)[0]]
        if len(v) == 0:
            continue
        idx = v[0] ## for legend
        cumsum = cumsums[idx]
        if is_first: ## only plot once.
            is_first = False
            fig = go.Figure(data=go.Scatter3d(x=cumsum[0],
                                              y=cumsum[1],
                                              z=cumsum[2],
                                              line=dict(
                                                  color=color_palette[i],
                                                  width=2
                                              ),
                                              legendgroup=str(cluster),
                                              showlegend=show_legend,
                                              name = str(cluster),
                                              marker_size = 150,
                                              mode='lines',
                                              **args
                                          ))
        else:
            fig.add_scatter3d(x=cumsum[0],
                              y=cumsum[1],
                              z=cumsum[2],
                              line=dict(
                                color=color_palette[i],
                                width=2
                            ),
                            legendgroup=str(cluster),
                            showlegend=show_legend,
                            name = str(cluster),
                            mode='lines',
                            **args
            )
        if sample_ratio < 1:
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)
        for idx in v[1:]:
            cumsum = cumsums[idx]
            fig.add_scatter3d(x=cumsum[0],
                              y=cumsum[1],
                              z=cumsum[2],
                              line=dict(
                                color=color_palette[i],
                                width=2
                            ),
                            name = str(cluster),
                            showlegend=False,
                            legendgroup=str(cluster),
                            mode='lines',
                            **args
            )
    fig.update_layout(
                    legend= {'itemsizing': 'constant'}, ## increase the point size in legend.
                    autosize=False,
                    width=figsize[1],
                    height=figsize[0],)

    if not return_fig:
        fig.show()
    if fig_path is not None:
        if fig_path.endswith(".html") or fig_path.endswith(".htm"):
            fig.write_html(fig_path)
        elif fig_path.endswith(".svg") or \
                fig_path.endswith(".pdf") or \
                fig_path.endswith(".eps") or \
                fig_path.endswith(".webp") or \
                fig_path.endswith(".png") or \
                fig_path.endswith(".jpg") or \
                fig_path.endswith(".jpeg"):
            module_check_install("kaleido")
            fig.write_image(fig_path, engine="kaleido")
    return fig if return_fig else None




def M_plot_trajectory_harmonic_lines(mat_coord_Hspace,
                                   cluster_list,
                                   retain_clusters=[],
                                   dims = [0,1],
                                   show_legend=True,
                                   legend_loc="center left",
                                   bbox_to_anchor=(1, 0.5),
                                   markerscale=4,
                                   ax = None,
                                   sample_ratio = 1,
                                   xylabel = True,
                                   color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                   **args):
    """
    Parameters
    ---------
    mat_coord_Hspace:
        hspace coordinates for each trajectory
    cluster_list:
        cluster_list for each trajectory
    ax:
        matplotlib axes
    show_legend:
        if show legend
    legend_loc:
        legend location
    bbox_to_anchor:
        for position of the legend
    markerscale:
        legend linewidth scale to larger or smaller
    ax:
        matplotlib axes
    sample_ratio:
        sample ratio for plotting
    xylabel:
        if show xy label
    color_palette:
        color palette for show cluster_list
    """
    assert(all(np.array(dims) < mat_coord_Hspace[0].shape[0])) ## dims is in the range of the dimension of the data
    assert(len(dims) ==2)
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    #print(retain_clusters)
    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset

    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)


    ax = ax or plt.gca()
    cumsums = list(map(lambda i: [np.cumsum(i[dims[0]]), np.cumsum(i[dims[1]])], mat_coord_Hspace))
    for i, cluster in enumerate(retain_clusters):
        #print(i, cluster)
        v = [i for i in np.where(cluster_list == cluster)[0]]
        if len(v) == 0:
            continue
        idx = v[0] ## for legend
        cumsum = cumsums[idx]

        #print(cumsum[0], cumsum[1], color_palette[i], cluster)
        ##(ci=None) to cancel confidence intervals computing
        sns.lineplot(x=cumsum[0], y=cumsum[1], color=color_palette[i], ax=ax, sort=False, label=cluster, ci=None,  estimator=None,n_boot=0, **args) #

        if sample_ratio < 1:
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)
        for idx in v[1:]:
            cumsum = cumsums[idx]
            sns.lineplot(x=cumsum[0], y=cumsum[1], color=color_palette[i], ax=ax, sort=False, ci=None,  estimator=None, n_boot=0, **args) #

    if xylabel:
        ax.set_xlabel(f"cumsum_{dims[0]}")
        ax.set_ylabel(f"cumsum_{dims[1]}")

    if show_legend:
        leg = ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
        for line in leg.get_lines():
            line.set_linewidth(markerscale)
    else:
        ax.get_legend().remove()


#endf plot_trajectory_harmonic_lines


def M_plot_trajectory_harmonic_points_3d(mat_coor_flatten_trajectory,
                                         cluster_list,
                                         retain_clusters=[],
                                         dims = [0,1,2],
                                         node_size=2,
                                         show_legend=False,
                                         figsize = (800,800),
                                         sample_ratio = 1,
                                         color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                         fig_path = None,
                                         return_fig = False,
                                         **args):
    """
    Parameters
    ---------
    mat_coor_flatten_trajectory:
        flatten trajectory matrix
    cluster_list:
        cluster_list for each trajectory
    show_legend:
        if show legend
    color_palette:
        color palette for show cluster_list
    **args:
        args for scatter
    """
    import plotly.graph_objects as go

    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset
    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)


    is_first = True
    for i, cluster in enumerate(retain_clusters):
        #print(i, cluster)
        v = [i for i in np.where(cluster_list == cluster)[0]]
        if len(v) == 0:
            continue
        idx = v[0] # for legend
        if sample_ratio < 1: ## need at least 1
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)

        df = pd.DataFrame({'x': [mat_coor_flatten_trajectory[ii][dims[0]] for ii in v],
                           'y': [mat_coor_flatten_trajectory[ii][dims[1]] for ii in v],
                           'z': [mat_coor_flatten_trajectory[ii][dims[2]] for ii in v],
                           'cluster': [cluster for ii in v],
                          })
        if is_first:
            is_first = False
            fig = go.Figure(data=[go.Scatter3d(x=df['x'],
                                               y=df['y'],
                                               z=df['z'],
                                               mode='markers',
                                               name=str(cluster),
                                               showlegend = show_legend,
                                               marker=dict(
                                                    size=node_size,
                                                    color=color_palette[i],                # set color to an array/list of desired values
                                                    **args)
                                               )])

        else:
            fig.add_scatter3d(x=df['x'],
                              y=df['y'],
                              z=df['z'],
                              mode='markers',
                              name=str(cluster),
                              showlegend = show_legend,
                              marker=dict(size=node_size,
                                          color=color_palette[i],                # set color to an array/list of desired values
                                          **args)

                             )


    fig.update_layout(
                legend= {'itemsizing': 'constant'}, ## increase the point size in legend.
                autosize=False,
                width=figsize[1],
                height=figsize[0],)

    if not return_fig:
        fig.show()
    if fig_path is not None:
        if fig_path.endswith(".html") or fig_path.endswith(".htm"):
            fig.write_html(fig_path)
        elif fig_path.endswith(".svg") or \
                fig_path.endswith(".pdf") or \
                fig_path.endswith(".eps") or \
                fig_path.endswith(".webp") or \
                fig_path.endswith(".png") or \
                fig_path.endswith(".jpg") or \
                fig_path.endswith(".jpeg"):
            module_check_install("kaleido")
            fig.write_image(fig_path, engine='kaleido')
    return fig if return_fig else None


def M_plot_trajectory_harmonic_points(mat_coor_flatten_trajectory,
                                    cluster_list,
                                    retain_clusters=[],
                                    dims = [0,1],
                                    label=True,
                                    labelsize=10,
                                    labelstyle='text',
                                    node_size=2,
                                    show_legend=False,
                                    legend_loc="center left",
                                    bbox_to_anchor=(1, 0.5),
                                    markerscale=4,
                                    ax = None,
                                    sample_ratio = 1,
                                    color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                    **args):
    """
    Parameters
    ---------
    mat_coor_flatten_trajectory:
        flatten trajectory matrix
    cluster_list:
        cluster_list for each trajectory
    label:
        if show label
    labelsize:
        labelsize
    labelstyle:
        options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    show_legend:
        if show legend
    legend_loc:
        legend location
    bbox_to_anchor:
        for position of the legend
    markerscale:
        legend marker scale to larger or smaller
    color_palette:
        color palette for show cluster_list
    **args:
        args for scatter
    """

    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    #print(retain_clusters)
    #print(set(cluster_list))
    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset
    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)

    ax = ax or plt.gca()

    if label:
        labeldf = pd.DataFrame(mat_coor_flatten_trajectory)[dims]
        labeldf.columns = ['x', 'y']
        labeldf['label'] = list(cluster_list)

    for i, cluster in enumerate(retain_clusters):
        #print(i, cluster)
        v = [i for i in np.where(cluster_list == cluster)[0]]
        if len(v) == 0:
            continue
        idx = v[0] # for legend
        if sample_ratio < 1: ## need at least 1
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)

        ax.scatter(mat_coor_flatten_trajectory[idx][dims[0]],
                   mat_coor_flatten_trajectory[idx][dims[1]],
                   color=color_palette[i],
                   s=node_size,
                   label=cluster, ## for legend
                   **args)
        for idx in v[1:]:
            ax.scatter(mat_coor_flatten_trajectory[idx][dims[0]],
                       mat_coor_flatten_trajectory[idx][dims[1]],
                       color=color_palette[i],
                       s=node_size,
                       **args)
            ax.set_xlabel(f"t-map_{dims[0]}")
            ax.set_ylabel(f"t-map_{dims[1]}")

        if label:
            if labelstyle=='text' or labelstyle == "color":
                ax.annotate(cluster,
                        labeldf.loc[labeldf['label']==cluster, ['x','y']].median(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="black" if labelstyle == "text" else color_palette[i])
            elif labelstyle == "box":
                ax.annotate(cluster,
                        labeldf.loc[labeldf['label']==cluster,['x','y']].median(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="white",
                        backgroundcolor=color_palette[i])
            else:
                print("warning, labelstyle is not correct, options: color, text, box")

    if show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, markerscale=markerscale)
#endf plot_trajectory_harmonic_points


def L_plot_eigen_line(values, n_eig=10, step_size=1, show_legend=True, ax=None, **args):
    """
    Parameters
    ---------
    values:
        eigenvalues list
    n_eig:
        number of eigen values to plot
    step_size:
        x-ticks step size
    ax:
        matplotlib ax
    **args:
        args for ax.plot
    """
    ax = ax or plt.gca()
    n_eig = min(n_eig, len(values))
    ax.plot(range(0,n_eig), values[0:n_eig], linestyle='--', marker='o', color='b', label='eigen value', **args)
    ax.set_xticks(range(0,n_eig, step_size))
    if show_legend:
        ax.legend()
#endf plot_eigen_line

