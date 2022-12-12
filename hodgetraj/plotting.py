import itertools
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Iterable, List, Optional, Set, Tuple, TypeVar

V = TypeVar('V')
def edges_on_path(path: List[V]) -> Iterable[Tuple[V, V]]:
    return zip(path, path[1:])

def nxdraw_group_legend(g,
                        layouts,
                        groups,
                        show_edges:bool=True,
                        show_legend:bool=True,
                        color_palette = sns.color_palette('tab10').as_hex(),
                        legend_loc="center left",
                        legend_size=6,
                        bbox_to_anchor=(1, 0.5),
                        markerscale =1,
                        label=True,
                        labelsize=10,
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

    if show_edges:
        nx.draw_networkx_edges(g, pos=layouts, ax=ax)
    for i, (k, v) in enumerate(d_colors.items()):
        name = k
        nodes = v
        nx.draw_networkx_nodes(g, pos=layouts, nodelist=nodes, ax=ax,
                               node_color=color_palette[i], label=rev_mapping[name], **args)
        if label:
            labeldf = pd.DataFrame(layouts).T
            labeldf.columns = ['x', 'y']
            labeldf['label'] = groups
            ax.annotate(d_group[k],
                        labeldf.loc[labeldf['label']==d_group[name],['x','y']].mean(),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=labelsize, weight='bold',
                        color="black")

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

