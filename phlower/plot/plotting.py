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

from ..util import get_uniform_multiplication, kde_eastimate, norm01

V = TypeVar('V')
def edges_on_path(path: List[V]) -> Iterable[Tuple[V, V]]:
    return zip(path, path[1:])

def plot_trajectory_harmonic_lines_3d(adata: AnnData,
                                      full_traj_matrix="full_traj_matrix",
                                      clusters = "trajs_clusters",
                                      evector_name = "X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                                      retain_clusters=[],
                                      figsize = (800,800),
                                      dims = [0,1,2],
                                      show_legend=True,
                                      sample_ratio = 0.1,
                                      color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                      **args):
    """
    Parameters
    ---------
    mat_coord_Hspace:
    cluster_list: cluster_list for each trajectory
    ax: matplotlib axes
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend linewidth scale to larger or smaller
    color_palette: color palette for show cluster_list
    """
    mat_coord_Hspace = M_create_matrix_coordinates_trajectory_Hspace(adata.uns[evector_name][0:max(dims)+1],
                                                                     adata.uns[full_traj_matrix])
    M_plot_trajectory_harmonic_lines_3d(mat_coord_Hspace,
                                        cluster_list = list(adata.uns[clusters]),
                                        retain_clusters=retain_clusters,
                                        dims = dims,
                                        show_legend = show_legend,
                                        sample_ratio = sample_ratio,
                                        color_palette = color_palette,
                                        **args)





def plot_trajectory_harmonic_lines(adata: AnnData,
                                   full_traj_matrix="full_traj_matrix",
                                   clusters = "trajs_clusters",
                                   evector_name = "X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                                   retain_clusters=[],
                                   dims = [0,1],
                                   show_legend=True,
                                   legend_loc="center left",
                                   bbox_to_anchor=(1, 0.5),
                                   markerscale=4,
                                   ax = None,
                                   sample_ratio = 0.1,
                                   color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                   **args):
    """
    Parameters
    ---------
    mat_coord_Hspace:
    cluster_list: cluster_list for each trajectory
    ax: matplotlib axes
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend linewidth scale to larger or smaller
    color_palette: color palette for show cluster_list
    """
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
                                     color_palette = color_palette,
                                     **args)



def plot_trajectory_harmonic_points_3d(adata: AnnData,
                                       full_traj_matrix_flatten="full_traj_matrix_flatten",
                                       clusters = "trajs_clusters",
                                       evector_name = "X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                                       retain_clusters=[],
                                       dims = [0,1],
                                       node_size=2,
                                       show_legend=False,
                                       figsize=(800,800),
                                       sample_ratio = 0.1,
                                       color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                       **args):
    """
    Parameters
    ---------
    mat_coor_flatten_trajectory:
    cluster_list: cluster_list for each trajectory
    label: if show label
    labelsize: labelsize
    labelstyle: options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend marker scale to larger or smaller
    color_palette: color palette for show cluster_list
    **args: args for scatter
    """
    mat_coor_flatten_trajectory = [adata.uns[evector_name][0:max(dims)+1, :] @ mat for mat in adata.uns[full_traj_matrix_flatten]]
    M_plot_trajectory_harmonic_points_3d(mat_coor_flatten_trajectory,
                                         cluster_list = list(adata.uns[clusters]),
                                         retain_clusters = retain_clusters,
                                         dims = dims,
                                         node_size = node_size,
                                         show_legend = show_legend,
                                         figsize = figsize,
                                         sample_ratio = sample_ratio,
                                         color_palette = color_palette,
                                         **args
                                         )




def plot_trajectory_harmonic_points(adata: AnnData,
                                    full_traj_matrix_flatten="full_traj_matrix_flatten",
                                    clusters = "trajs_clusters",
                                    evector_name = "X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
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
    Parameters
    ---------
    mat_coor_flatten_trajectory:
    cluster_list: cluster_list for each trajectory
    label: if show label
    labelsize: labelsize
    labelstyle: options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend marker scale to larger or smaller
    color_palette: color palette for show cluster_list
    **args: args for scatter
    """
    mat_coor_flatten_trajectory = [adata.uns[evector_name][0:max(dims)+1, :] @ mat for mat in adata.uns[full_traj_matrix_flatten]]
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
                             graph_name = "X_dm_ddhodge_g",
                             layout_name = "X_dm_ddhodge_g",
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
            pos=nx.get_node_attributes(adata.uns[fate_tree], 'pos'),
            node_size=node_size,
            labels=labels,
            with_labels=with_labels,
            ax=ax)
#endf plot_fate_tree_embedding

def plot_stream_tree_embedding(adata: AnnData,
                             graph_name = "X_dm_ddhodge_g",
                             layout_name = "X_dm_ddhodge_g",
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
            pos=nx.get_node_attributes(adata.uns[stream_tree], 'pos'),
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
    layout_prog: may ‘dot’, ‘twopi’, ‘fdp’, ‘sfdp’, ‘circo’
    """
    ax = plt.gca() if ax is None else ax
    pos =nx.nx_pydot.graphviz_layout(adata.uns[fate_tree], prog=layout_prog)
    nx.draw(adata.uns[fate_tree], pos, with_labels=with_labels, ax=ax, **args)


def plot_density_grid(adata: AnnData,
                      graph_name = "X_dm_ddhodge_g_triangulation_circle",
                      layout_name = "X_dm_ddhodge_g",
                      cluster_name = "trajs_clusters",
                      trajs_name = "knn_trajs",
                      retain_clusters=[],
                      sample_n=10000,
                      figsize=(20,16),
                      title_prefix='cluster_',
                      bg_alpha = 0.5,
                      node_size = 2,
                      **args
                      ):

    if graph_name not in adata.uns.keys():
        raise ValueError("graph_name not in adata.uns.keys()")
    if layout_name not in adata.obsm.keys():
        raise ValueError("layout_name not in adata.obsm.keys()")
    if cluster_name not in adata.uns.keys():
        raise ValueError("cluster_name not in adata.uns.keys()")
    if trajs_name not in adata.uns.keys():
        raise ValueError("trajs_name not in adata.uns.keys()")

    G_plot_density_grid(adata.uns[graph_name],
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
                    evalue_name="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_value",
                    n_eig=10,
                    step_size=1,
                    show_legend=True,
                    ax=None,
                    **args):
    L_plot_eigen_line(adata.uns[evalue_name],
                      n_eig=n_eig,
                      step_size=step_size,
                      show_legend=show_legend,
                      ax=ax,
                      **args)
#endf

def plot_traj(adata: AnnData,
              graph_name: str = 'X_dm_ddhodge_g_triangulation',
              layout_name: str='X_dm_ddhodge_g',
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

def nxdraw_holes(adata: AnnData,
                 graph_name: str='X_dm_ddhodge_g_triangulation_circle',
                 layout_name: str='X_dm_ddhodge_g_triangulation_circle',
                 evector_name: str="X_dm_ddhodge_g_triangulation_circle_L1Norm_decomp_vector",
                 title = "",
                 edge_value= [],
                 vector_dim=0,
                 font_size=0,
                 node_size=0.1,
                 width=1,
                 edge_cmap=plt.cm.RdBu_r,
                 with_labels = False,
                 with_potential = ['X_dm_ddhodge', 'u'],
                 flip = False,
                 is_norm=False,
                 is_abs = False,
                 is_arrow = True,
                 ax = None,
                 **args):

    ax = ax or plt.gca()
    H = adata.uns[evector_name][vector_dim]
    if is_norm:
        H = np.abs(norm01(H)) if is_abs else norm01(H)
    if len(edge_value)>0:
        H = np.abs(edge_value) if is_abs else edge_value

    elist = adata.uns[graph_name].edges()
    elist_set = set(list(elist))
    if with_potential is not None and  len(with_potential) == 2:
        u=nx.get_node_attributes(adata.uns[graph_name], with_potential[1])
        direction = [-1 if (u[e]-u[a])>0 else 1 for a,e in elist]
        H = [i*j for i,j in zip(H, direction)]

    gg = adata.uns[graph_name].to_directed()
    for edge in list(gg.edges()):
        if edge in elist_set:
            continue
        gg.remove_edge(edge[0], edge[1])

    if flip:
        H = [i*-1 for i in H]
    nx.draw_networkx(gg if is_arrow else gg.to_undirected(),
                     adata.obsm[graph_name],
                     edge_color=H,
                     node_size=node_size,
                     width=width,
                     edge_cmap=edge_cmap,
                     with_labels=with_labels,
                     ax=ax,
                     **args)

    if title:
        ax.set_title(title)
#endf nxdraw_Holes


def nxdraw_score(adata: AnnData,
                 graph_name:str = "X_dm_ddhodge_g",
                 layout_name:str = "X_dm_ddhodge_g",
                 color:str = "u",
                 colorbar:bool = True,
                 directed:bool = False,
                 font_size:float = 0,
                 cmap = plt.cm.get_cmap('viridis'),
                 **args
    ):
    u_color = None
    if color in set(chain.from_iterable(d.keys() for *_, d in adata.uns['X_dm_ddhodge_g'].nodes(data=True))):
        u_color = np.fromiter(nx.get_node_attributes(adata.uns['X_dm_ddhodge_g'], color).values(), dtype='float')
    elif color in adata.obs:
        u_color = adata.obs[color].values

    ##TODO assert digital

    if directed:
        nx.draw_networkx(adata.uns[graph_name], adata.obsm[layout_name], node_color=u_color, cmap=cmap, font_size=font_size, **args)
    else:
        nx.draw_networkx(adata.uns[graph_name].to_undirected(), adata.obsm[layout_name], cmap=cmap, node_color=u_color, font_size=font_size, **args)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = min(u_color), vmax=max(u_color)))
        sm._A = []
        plt.colorbar(sm)
#endf nxdraw_score



def nxdraw_group(adata: AnnData,
                 graph_name:str = 'X_dm_ddhodge_g',
                 layout_name: str = 'X_dm_ddhodge_g',
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
                 ax = None,
                 **args):

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
                   labelsize=labelsize,
                   labelstyle=labelstyle,
                   directed = directed,
                   ax=ax,
                   **args)


#endf nxdraw_group

def plot_triangle_density(adata: AnnData,
                          graph_name: str = 'X_dm_ddhodge_g',
                          layout_name: str = 'X_dm_ddhodge_g',
                          node_size=10,
                          ax=None,
                          cmap = plt.get_cmap("jet"),
                          colorbar = True,
                          **args):

    G_plot_triangle_density(adata.uns[graph_name], adata.obsm[layout_name], node_size=node_size, ax=ax, cmap=cmap, colorbar=colorbar, **args)
#endf plot_triangle_density

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
                 labelstyle='text',
                 directed=False,
                 ax = None,
                 **args):
    """
    Parameters
    ---------
    g: networkx graph
    layouts: layouts dict or array
    groups: groups list
    show_edges: if show edges
    color_palette: color palette for show groups
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend marker scale to larger or smaller
    label: if show label
    labelsize: labelsize
    labelstyle: options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
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
            nx.draw_networkx_edges(g, pos=layouts, ax=ax)
        else:
            nx.draw_networkx_edges(g.to_undirected(), pos=layouts, ax=ax)

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
            ax.legend(loc=legend_loc,  bbox_to_anchor=bbox_to_anchor, markerscale=markerscale)



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
                ) -> None:

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
        nx.draw_networkx_edges(graph,
                               node_positions,
                               ax=ax,
                               edgelist=list(edges_on_path(trajectory)),
                               node_size=10,
                               width=edge_width,
                               edge_color=color,
                               arrows=True,
                               arrowstyle='->')

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
    g: networkx graph
    layouts: layouts dict or array
    cmap: matplotlib.colors.LinearSegmentedColormap
    colorbar: if show colorbar
    **args: parameters of networkx.draw
    """

    ax = ax or plt.gca()
    values = nx.triangles(g)
    n_color = np.asarray([values[n] for n in g.nodes()])
    nx.draw(g, layouts, node_color=n_color, node_size=node_size, ax=ax, cmap=cmap, **args)
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(n_color), vmax=max(n_color)))
        sm.set_array([])
        plt.colorbar(sm)
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
    cluster_list: cluster labels for each point
    layouts: embeddings, shape should be nx2
    retain_clusters: which clusters to plot
    node_size: node size
    label: if add labels
    labelsize: size of labels
    labelstyle: options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    ax: matplotlib ax
    show_legend: if show_legend
    legend_loc: legend location
    bbox_to_anchor: tune of legend position
    markerscale: legend markerscale
    facecolor: plt background
    **args: parameters of ax.scatter
    """
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    if len(cluster_list)==0 or embedding is None:
        print("Error: cluster_list and embedding should be not None!")
        return
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
                      bg_alpha = 0.5,
                      node_size = 2,
                      **args
                      ):

    """
    Parameters
    ---------
    G: networkx graph
    layouts: layouts dict or array
    cluster_list: cluster label list
    retain_clusters: only show clusters in retain_clusters
    sample_n: sample the for the kde estimate to save time
    figsize: figure size tuple
    title_prefix: prefix for each subplot
    node_size: density node size
    """
    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)

    assert(set(retain_clusters).issubset(set(cluster_list))) ## is subset
    cluster_n = len(retain_clusters)
    assert(cluster_n > 0)

    if not isinstance(cluster_list, np.ndarray):
        cluster_list = np.array(cluster_list)

    if not isinstance(traj_list, np.ndarray):
        traj_list = np.array(traj_list)

    r,c = get_uniform_multiplication(cluster_n)
    r = r - 1 if (r-1)*c == cluster_n else r
    fig, axes = plt.subplots(r,c, sharex=False, sharey=False,)
    fig.set_size_inches(figsize[0], figsize[1])
    for a, cluster in enumerate(retain_clusters):
        i = int(a/c)
        j = a%c
        idx = [i for i in np.where(cluster_list == cluster)[0]]
        cdic = kde_eastimate(traj_list[idx], layouts, sample_n)
        x = cdic['x']
        y = cdic['y']
        z = cdic['z']
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

    return fig, axes



def M_plot_trajectory_harmonic_lines_3d(mat_coord_Hspace,
                                        cluster_list,
                                        retain_clusters=[],
                                        dims = [0,1,2],
                                        figsize = (800, 800),
                                        show_legend=True,
                                        sample_ratio = 1,
                                        color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                        **args):
    """
    Parameters
    ---------
    mat_coord_Hspace:
    cluster_list: cluster_list for each trajectory
    ax: matplotlib axes
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend linewidth scale to larger or smaller
    color_palette: color palette for show cluster_list
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
                                              mode='lines'
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
                            mode='lines'
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
                            mode='lines'
            )
    fig.update_layout(
                legend= {'itemsizing': 'constant'}, ## increase the point size in legend.
                autosize=False,
                width=figsize[1],
                height=figsize[0],)
    fig.show()


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
                                   color_palette = sns.color_palette(cc.glasbey, n_colors=50).as_hex(),
                                   **args):
    """
    Parameters
    ---------
    mat_coord_Hspace:
    cluster_list: cluster_list for each trajectory
    ax: matplotlib axes
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend linewidth scale to larger or smaller
    color_palette: color palette for show cluster_list
    """
    assert(all(np.array(dims) < mat_coord_Hspace[0].shape[0])) ## dims is in the range of the dimension of the data
    assert(len(dims) >=2)
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
        sns.lineplot(x=cumsum[0], y=cumsum[1], color=color_palette[i], ax=ax, sort=False, label=cluster, **args) #

        if sample_ratio < 1:
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)
        for idx in v[1:]:
            cumsum = cumsums[idx]
            sns.lineplot(x=cumsum[0], y=cumsum[1], color=color_palette[i], ax=ax, sort=False, **args) #

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
                                         **args):
    """
    Parameters
    ---------
    mat_coor_flatten_trajectory:
    cluster_list: cluster_list for each trajectory
    show_legend: if show legend
    color_palette: color palette for show cluster_list
    **args: args for scatter
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

    fig.show()


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
    cluster_list: cluster_list for each trajectory
    label: if show label
    labelsize: labelsize
    labelstyle: options: color,text, box. same color as nodes if use `color`, black if use `text`, white color with box if use `box`
    show_legend: if show legend
    legend_loc: legend location
    bbox_to_anchor: for position of the legend
    markerscale: legend marker scale to larger or smaller
    color_palette: color palette for show cluster_list
    **args: args for scatter
    """

    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
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
    values: eigenvalues list
    n_eig: number of eigen values to plot
    step_size: x-ticks step size
    ax: matplotlib ax
    **args: args for ax.plot
    """
    ax = ax or plt.gca()
    n_eig = min(n_eig, len(values))
    ax.plot(range(1,n_eig+1), values[0:n_eig], linestyle='--', marker='o', color='b', label='eigen value', **args)
    ax.set_xticks(range(1,n_eig+1, step_size))
    if show_legend:
        ax.legend()
#endf plot_eigen_line
