import itertools
import numpy as np
import colorcet as cc
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from collections import defaultdict
from typing import Iterable, List, Optional, Set, Tuple, TypeVar

from ..util import get_uniform_multiplication, kde_eastimate

V = TypeVar('V')
def edges_on_path(path: List[V]) -> Iterable[Tuple[V, V]]:
    return zip(path, path[1:])


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
                          show_colorbar = True,
                          **args):

    """
    Parameters
    ---------
    g: networkx graph
    layouts: layouts dict or array
    cmap: matplotlib.colors.LinearSegmentedColormap
    show_colorbar: if show colorbar
    **args: parameters of networkx.draw
    """

    ax = ax or plt.gca()
    values = nx.triangles(g)
    n_color = np.asarray([values[n] for n in g.nodes()])
    nx.draw(g, layouts, node_color=n_color, node_size=node_size, ax=ax, cmap=cmap, **args)
    if show_colorbar:
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

    ax = ax or plt.gca()

    if label:
       labeldf = pd.DataFrame(embedding)
       labeldf.columns = ['x', 'y']
       labeldf['label'] = list(cluster_list)

    cluster_n = len(set(cluster_list))
    ax.set_facecolor(facecolor)
    for i, x in enumerate(retain_clusters):
        idx = [i for i in np.where(np.array(cluster_list) == x)[0]]
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



def plot_density_grid(G,
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

    r,c = get_uniform_multiplication(cluster_n)
    r = r - 1 if (r-1)*c == cluster_n else r
    fig, axes = plt.subplots(r,c, sharex=False, sharey=False,)
    fig.set_size_inches(figsize[0], figsize[1])
    for a, cluster in enumerate(retain_clusters):
        i = int(a/c)
        j = a%c
        idx = [i for i in np.where(np.array(cluster_list) == cluster)[0]]
        cdic = kde_eastimate(np.array(traj_list)[idx], layouts, sample_n)
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

def plot_trajectory_harmonic_lines(mat_coord_Hspace,
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


    ax = ax or plt.gca()
    cumsums = list(map(lambda i: [np.cumsum(i[dims[0]]), np.cumsum(i[dims[1]])], mat_coord_Hspace))
    for i, cluster in enumerate(retain_clusters):
        #print(i, cluster)
        v = [i for i in np.where(np.array(cluster_list) == cluster)[0]]
        idx = v[0] ## for legend
        cumsum = cumsums[idx]
        sns.lineplot(x=cumsum[0], y=cumsum[1], color=color_palette[i], ax=ax, sort=False, label=cluster, **args)

        if sample_ratio < 1:
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)
        for idx in v[1:]:
            cumsum = cumsums[idx]
            sns.lineplot(x=cumsum[0], y=cumsum[1], color=color_palette[i], ax=ax, sort=False, **args)

    if show_legend:
        leg = ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
        for line in leg.get_lines():
            line.set_linewidth(markerscale)
    else:
        ax.get_legend().remove()


#endf plot_trajectory_harmonic_lines

def plot_trajectory_harmonic_points(mat_coor_flatten_trajectory,
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

    ax = ax or plt.gca()

    if label:
        labeldf = pd.DataFrame(mat_coor_flatten_trajectory)[dims]
        labeldf.columns = ['x', 'y']
        labeldf['label'] = list(cluster_list)

    for i, cluster in enumerate(retain_clusters):
        #print(i, cluster)
        v = [i for i in np.where(np.array(cluster_list) == cluster)[0]]
        idx = v[0] # for legend
        if sample_ratio < 1: ## need at least 1
            np.random.seed(2022)
            v = np.random.choice(v, max(int(len(v)*sample_ratio), 1), replace=False)

        ax.scatter(mat_coor_flatten_trajectory[idx][0],
                       mat_coor_flatten_trajectory[idx][1],
                       color=color_palette[i],
                       s=node_size,
                       label=cluster, ## for legend
                       **args)
        for idx in v[1:]:
            ax.scatter(mat_coor_flatten_trajectory[idx][0],
                       mat_coor_flatten_trajectory[idx][1],
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


def plot_eigen_line(values, n_eig=10, step_size=1, show_legend=True, ax=None, **args):
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

