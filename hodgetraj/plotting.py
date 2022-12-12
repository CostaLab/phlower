import itertools
import numpy as np
import colorcet as cc
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Iterable, List, Optional, Set, Tuple, TypeVar

V = TypeVar('V')
def edges_on_path(path: List[V]) -> Iterable[Tuple[V, V]]:
    return zip(path, path[1:])

def nxdraw_group(g,
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
        labeldf = pd.DataFrame(layouts).T
        labeldf.columns = ['x', 'y']
        labeldf['label'] = groups

    if show_edges:
        nx.draw_networkx_edges(g, pos=layouts, ax=ax)
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



def plot_traj(graph: nx.Graph,
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
                               width=2,
                               edge_color=color,
                               arrows=True,
                               arrowstyle='->')

    if hole_centers is not None:
        ax.scatter(x=hole_centers[:, 0], y=hole_centers[:, 1], marker='x')



def plot_triangle_density(g:nx.Graph,
                          layouts,
                          node_size=10,
                          ax=None,
                          cmap = plt.get_cmap("jet"),
                          show_colorbar = True,
                          **args
                          ):

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
       labeldf['label'] = cluster_list

    cluster_n = len(set(cluster_list))
    ax.set_facecolor(facecolor)
    for i, x in enumerate(set(cluster_list)):
        idx = [i for i in np.where(cluster_list == x)[0]]
        ax.scatter(x=embedding[idx, 0], y=embedding[idx, 1], c = color_palette[i], s=node_size, **args)
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
        ax.legend(set(cluster_list), loc=legend_loc, bbox_to_anchor=bbox_to_anchor, markerscale=markerscale)


