import numpy as np
import copy
import random
import networkx as nx
from anndata import AnnData
from scipy.sparse import linalg, csr_matrix
from scipy.spatial import Delaunay,distance
from ..util import tuple_increase, top_n_from, is_in_2sets, is_node_attr_existing


def construct_trucated_delaunay(adata:AnnData,
                                graph_name:str='X_dm_ddhodge_g',
                                layout_name:str='X_dm_ddhodge_g',
                                trunc_quantile:float=0.75,
                                trunc_times:float=3,
                                iscopy:bool=False,
                                ):
    if iscopy:
        adata = adata.copy()

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")

    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")


    edges = truncated_delaunay(adata.obsm[layout_name], trunc_quantile=trunc_quantile, trunc_times=trunc_times)
    adata.uns[f'{graph_name}_triangulation'] = reset_edges(adata.uns[graph_name], edges, keep_old=False)

    return adata if iscopy else None
#end construct_trucated_delaunay

def construct_circle_delaunay(adata:AnnData,
                              graph_name:str='X_dm_ddhodge_g_triangulation',
                              layout_name:str='X_dm_ddhodge_g',
                              cluster_name:str='group',
                              quant=0.1,
                              node_attr='u',
                              start_n=5,
                              end_n = 5,
                              separate_ends_triangle = False,
                              random_seed = 2022,
                              calc_layout:bool = False,
                              iscopy:bool=False,
                              ):
    if iscopy:
        adata = adata.copy()

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")
    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")
    if cluster_name not in adata.obs:
        raise ValueError(f"{cluster_name} not in adata.obs")
    if not is_node_attr_existing(adata.uns[graph_name], node_attr):
        raise ValueError(f"{node_attr} not in adata.uns[{graph_name}]")

    if quant >=1 or quant <=0:
        raise ValueError(f"quant:{quant} should between (0,1)")

    if start_n <1 or start_n >= len(adata.obs.index) :
        raise ValueError(f"start_n:{start_n} should between (1, {len(adata.obs.index)})")
    if end_n <1 or end_n >= len(adata.obs.index) :
        raise ValueError(f"end_n:{end_n} should between (1, {len(adata.obs.index)})")


    layouts = adata.obsm[layout_name]
    group = adata.obs[cluster_name]
    adata.uns[f'{graph_name}_circle'] = connect_starts_ends_with_Delaunay(adata.uns[graph_name],
                                                                          layouts,
                                                                          group,
                                                                          quant=quant,
                                                                          node_attr=node_attr,
                                                                          start_n=start_n,
                                                                          end_n=end_n,
                                                                          separate_ends_triangle=separate_ends_triangle,
                                                                          random_seed=random_seed)

    if calc_layout:
        pydot_layouts = nx.nx_pydot.graphviz_layout(adata.uns[f"{graph_name}_circle"])
        adata.obsm[f'{graph_name}_circle'] = np.array([pydot_layouts[i] for i in range(len(pydot_layouts))])

    return adata if iscopy else None
#endf construct_circle_delaunay


def truncated_delaunay(position, trunc_quantile=0.75, trunc_times=3):
    """
    delaunay on a layout, then remove edges > trunc_quantile * trunc_times

    Parameters
    ---------
    position: 2D pd.array for the running of delaunay
    trunc_quantile: quantile value for the tunc of the Delaunay output
    trunc_times: distance of trunc_quantile x trunc_times will be removed
    """
    if trunc_quantile >=1 or trunc_quantile <=0:
        raise ValueError(f"trunc_quantile:{trunc_quantile} should between (0,1)")


    if type(position) == dict:
        position = np.array([position[x] for x in range(max(position.keys()) + 1)])
    tri = Delaunay(position)
    ti = tuple_increase
    tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
    tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
    edges_distance = [distance.euclidean(tuple(position[a]),tuple(position[b])) for (a,b) in tri_edges]
    threshold = np.quantile(edges_distance, trunc_quantile) * trunc_times
    keep_edges = [tri_edges[i] for i in range(len(tri_edges)) if edges_distance[i] < threshold]
    return keep_edges



def truncate_graph(G, layouts , trunc_quantile=0.75, trunc_times=3):
    """
    truncated the edges due to its long distance on the layouts

    Parameters
    ---------
    G: networkx.Graph
    layouts: layouts
    trunc_quantile:
    trunc_times:

    """
    trunc_quantile=0.75; trunc_times=3
    if type(layouts) == dict:
        layouts_list = np.array([layouts[x] for x in range(max(layouts.keys()) + 1)])
    edges = list(G.edges())

    edges_distance = [distance.euclidean(tuple(layouts_list[a]),tuple(layouts_list[b])) for (a,b) in edges]
    threshold = np.quantile(edges_distance, trunc_quantile) * trunc_times
    keep_edges = [edges[i] for i in range(len(edges)) if edges_distance[i] < threshold]
    nG = nx.create_empty_copy(G)
    nG.add_edges_from(keep_edges)
    return nG
#endf


def connect_starts_ends_with_Delaunay(g,
                                      layouts,
                                      group,
                                      quant=0.1,
                                      node_attr='u',
                                      start_n=5,
                                      end_n = 5,
                                      separate_ends_triangle = False,
                                      random_seed = 2022
                                      ):
    """
    untangle the connections between starts and ends generated by delauney.
    use the group information, keep only connnect within each group.
    this still produce isolated points.

    Parameters
    -------
    g: the graph generated by delaunay
    group: groups of each cell
    addedges: use which to extract all start celltypes and end celltyps.
    layouts: Delaunay layouts, for reconstructing delaunay on each start and end cluster
    start_n:
    end_n:
    separate_ends_triangle:
    random_seed:

    Return
    -----
    G_con: connected Delaunay graph for L1 holes
    """
    random.seed(random_seed)

    #import pdb
    #pdb.set_trace()


    values = np.fromiter(nx.get_node_attributes(g, node_attr).values(), dtype=np.float)

    n=len(g.nodes())
    o=np.sort(values)
    early=np.where(values<=o[round(n*quant)])[0]
    later=np.where(values>=o[round(n*(1-quant))])[0]

    #node_attr='u'
    #start_n = 10
    #end_n = 10


    if start_n <=0 or end_n <=0:
        raise ValueError("start_n and end_n must be positive")
    ## might be networkx.Graph layouts dict, convert to list
    if type(layouts) == dict:
        layouts = np.array([layouts[x] for x in range(max(layouts.keys()) + 1)])
    if not isinstance(layouts, np.ndarray):
        layouts = np.array(layouts)

    G_ae = copy.deepcopy(g)
    starts = early
    ends = later
    u = nx.get_node_attributes(g, node_attr)

    start_cts = list(set(group[starts]))
    end_cts = list(set(group[ends]))
    print("start clusters ", start_cts)
    print("end clusters ", end_cts)
    start_nodes = np.concatenate([np.where(np.array(group) == start_ct)[0] for start_ct in start_cts]).ravel()
    n_start_nodes = top_n_from(start_nodes, u, min(start_n, len(start_nodes)), largest=False)

    end_nodes_arr = []
    for i in range(len(end_cts)):
        end_nodes_arr.append([i for i in np.where(np.array(group) == end_cts[i])[0] if i in set(ends)])

    end_nodes_sets = [set(x) for x in end_nodes_arr]
    #print(end_nodes_arr)
    #print(end_n)
    n_end_nodes_arr = [top_n_from(arr, u, min(end_n, len(arr)), largest=True) for arr in end_nodes_arr]
    n_end_nodes= [y for x in end_nodes_arr for y in x]

    #for n_end_nodes in end_nodes_arr:
    #    G_ae.add_edges_from(zip(n_start_nodes, random.choices(list(n_end_nodes), k=end_n)))
    #    G_ae.add_edges_from(zip(n_end_nodes, random.choices(list(n_start_nodes), k=start_n)))
    ti = tuple_increase
    if separate_ends_triangle: ## each ends create triangles with the starts
        for i in range(len(n_end_nodes_arr)):
            selected_nodes = list(n_start_nodes) + list(n_end_nodes_arr[i])
            tri = Delaunay(layouts[selected_nodes])
            tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
            tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
            tri_edges = [(selected_nodes[x], selected_nodes[y]) for (x,y) in tri_edges]
            #print(len(tri_edges))
            G_ae.add_edges_from(tri_edges)
    else:
        selected_nodes = list(n_start_nodes) + list(n_end_nodes)
        tri = Delaunay(np.array(layouts)[selected_nodes])
        tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
        tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
        tri_edges = [(selected_nodes[x], selected_nodes[y]) for (x,y) in tri_edges]
        print(len(tri_edges))
        tri_edges = [(x,y) for (x,y) in tri_edges if not is_in_2sets(x,y, end_nodes_sets)]
        print(len(tri_edges))
        ## filtering, ends should not be together.
        G_ae.add_edges_from(tri_edges)

    return G_ae
#endf connect_starts_ends_with_Delaunay




def reset_edges(g:nx.Graph, edges, keep_old=False) -> nx.Graph:
    """
    Replace all edges with new edges

    Parameters
    ----------
    g: graph
    edges: new edges to replace edges of g
    keep_old: if keep old edges of the graph
    """
    #ti=tuple_increase
    #edges = [ti(x[0], x[1]) for x in edges]
    ti = tuple_increase

    ng = nx.create_empty_copy(g)
    ng = nx.Graph(nx.to_undirected(ng))
    if keep_old:
        diffusion_knn_edge = [ti(x[0], x[1]) for x in g.edges()]
        merged_edges = diffusion_knn_edge+ edges
        merged_edges = list(set(merged_edges))
        ng.add_edges_from(merged_edges)
    else:
        ng.add_edges_from(edges)
    return ng





def circumcircle_radius(a, b, c):
    """
    Calculates the radius of the circumcircle of a triangle given the distances between three points.
    Input: a, b, and c are the distances between the three points as scalars.
    Output: a scalar representing the radius of the circumcircle.
    """
    import math
    s = (a + b + c) / 2
    R = a*b*c / (4*math.sqrt(s*(s-a)*(s-b)*(s-c)))
    return R

def find_tri_combinations(G):
    import itertools
    triangles = []
    for nodes in itertools.combinations(G.nodes(), 3):
        # Check if the three nodes form a triangle
        if G.has_edge(nodes[0], nodes[1]) and G.has_edge(nodes[1], nodes[2]) and G.has_edge(nodes[2], nodes[0]):
            triangles.append(nodes)
    return triangles


def distance_tri(e_df, distance, *tri):
    a,b,c = tri
    #idxs =
    idx1 = e_df.index[(e_df.source == a) & (e_df.target == b)][0] if len(e_df.index[(e_df.source == a) & (e_df.target == b)])>0 else e_df.index[(e_df.source == b) & (e_df.target == a)][0]
    idx2 = e_df.index[(e_df.source == a) & (e_df.target == c)][0] if len(e_df.index[(e_df.source == a) & (e_df.target == c)])>0 else e_df.index[(e_df.source == c) & (e_df.target == a)][0]
    idx3 = e_df.index[(e_df.source == b) & (e_df.target == c)][0] if len(e_df.index[(e_df.source == b) & (e_df.target == c)])>0 else e_df.index[(e_df.source == c) & (e_df.target == b)][0]
    d1 = e_df.iloc[idx1][distance]
    d2 = e_df.iloc[idx2][distance]
    d3 = e_df.iloc[idx3][distance]

    return float(d1),float(d2),float(d3)

def edges_tri(e_df, *tri):
    a,b,c = tri
    #idxs =
    idx1 = e_df.index[(e_df.source == a) & (e_df.target == b)][0] if len(e_df.index[(e_df.source == a) & (e_df.target == b)])>0 else e_df.index[(e_df.source == b) & (e_df.target == a)][0]
    idx2 = e_df.index[(e_df.source == a) & (e_df.target == c)][0] if len(e_df.index[(e_df.source == a) & (e_df.target == c)])>0 else e_df.index[(e_df.source == c) & (e_df.target == a)][0]
    idx3 = e_df.index[(e_df.source == b) & (e_df.target == c)][0] if len(e_df.index[(e_df.source == b) & (e_df.target == c)])>0 else e_df.index[(e_df.source == c) & (e_df.target == b)][0]

    e1 = int(e_df.iloc[idx1]['source']), int(e_df.iloc[idx1]['target'])
    e2 = int(e_df.iloc[idx2]['source']), int(e_df.iloc[idx2]['target'])
    e3 = int(e_df.iloc[idx3]['source']), int(e_df.iloc[idx3]['target'])


    return e1, e2, e3


def mean_tri_coor(coor, *tri):
    a,b,c = tri
    #idxs =
    x = np.mean([coor[a], coor[b], coor[c]], axis=0)

    return x
