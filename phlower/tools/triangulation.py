import numpy as np
import copy
import random
import networkx as nx
from anndata import AnnData
from scipy.sparse import linalg, csr_matrix
from scipy.spatial import Delaunay,distance
from ..util import tuple_increase, top_n_from, is_in_2sets, is_node_attr_existing
from .graphconstr import adjedges
from gudhi import SimplexTree
import gudhi as gh

def count_features_at_radius(adata, radius, with_holes=False):
    """
    Count Topological data analysis, 0-dimensional and 1-dimensional features at a given radius

    Parameters
    -----------
    adata: AnnData
        an Annodata object
    radius: float
        radius threshold
    with_holes:
        Which to use the triangulated graph with/without holeds
    """

    num_0D = 0  # Connected components

    ## select persitence
    if with_holes:
        simplex_tree = adata.uns['persistence']['simplextree_tri_ae']
        num_0D = nx.number_connected_components(adata.uns['X_pca_ddhodge_g_triangulation_circle'])
    else:
        simplex_tree = adata.uns['persistence']['simplextree_tri']
        num_0D = nx.number_connected_components(adata.uns['X_pca_ddhodge_g_triangulation'])

    num_1D = 0  # Loops/holes

    for dim, (birth, death) in simplex_tree.persistence():
        if birth <= radius < death or (death == float('inf') and birth <= radius):
            if dim == 0:
                pass
                #num_0D += 1  # Count connected components
            elif dim == 1:
                num_1D += 1  # Count holes
    return num_0D, num_1D
#endf count_features_at_radius

def construct_delaunay_persistence(adata: AnnData,
                                   graph_name:str=None,
                                   layout_name:str=None,
                                   filter_ratio:float=1.2,
                                   min_persistence=0.1,
                                   cluster_name:str='group',
                                   circle_quant=0.1,
                                   node_attr='u',
                                   start_n=5,
                                   end_n = 5,
                                   random_seed = 2022,
                                   calc_layout:bool = False,
                                   iscopy:bool=False,
        ):
    """
    reimplement the function of construct_delaunay by using persistence analysis filtering.

    1. first delaunay triangulation on graph.
    2. check barcode0 of different radius of from the triangulated graph
    3. when there's only one connected component move a litte bit forward.
    4. perform the filtering.
    5. connect starts and the ends.
    6. store the SimplexTree information for plotting

    Parameters
    ------------
    adata: AnnData
        an Annodata object
    graph_name: str
        the graph name in adata.uns
    layout_name: str
        the layout name in adata.obsm
    filter_ratio: float
        the cut-off threshold for triangulation, default 1.2
    min_persistence: float
        minimum value for simplextree persistence calculating
    circle_quant: float
        quantile for connecting start and end points
    node_attr: str
        node attribute for connecting
    start_n: int
        number of start points
    end_n: int
        number of end points
    random_seed: int
        random seed for selecting start and end points
    calc_layout: bool
        whether to calculate layout for the delaunay graph
    iscopy: bool
        whether to return a copy of adata
    """
    adata = adata.copy if iscopy else adata
    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")

    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")
    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")
    if cluster_name not in adata.obs:
        raise ValueError(f"{cluster_name} not in adata.obs")
    if not is_node_attr_existing(adata.uns[graph_name], node_attr):
        raise ValueError(f"{node_attr} not in adata.uns[{graph_name}]")

    if start_n <1 or start_n >= len(adata.obs.index) :
        raise ValueError(f"start_n:{start_n} should between (1, {len(adata.obs.index)})")
    if end_n <1 or end_n >= len(adata.obs.index) :
        raise ValueError(f"end_n:{end_n} should between (1, {len(adata.obs.index)})")

    g = adata.uns[graph_name]
    layouts = adata.obsm[graph_name]
    ## 1. only generate gae edges for connecting starts and ends
    ae_edges = connect_starts_ends_with_Delaunay_edges(g,
                                                       layouts,
                                                       adata.obs[cluster_name],
                                                       quant=circle_quant,
                                                       node_attr=node_attr,
                                                       start_n=start_n,
                                                       end_n = end_n,
                                                       random_seed = random_seed)
    ## 2. create filtration_list and simplextree
    tri_edge_list, filtration_list, simplex_tree = simplextree(layouts)
    tri_edge_list_ae, filtration_list_ae, simplex_tree_ae = simplextree_ae(reset_edges(g, ae_edges), layouts)


    ## 3. persistence analysis
    gtri = g.copy()
    pers = simplex_tree.persistence(min_persistence=min_persistence)
    #print(pers)
    barcode0 = [b for b in pers if b[0] == 0]
    if len(pers) <=1:
        print("Warning: too large min_persistence, should decrease")
    #print(barcode0)
    maxx = barcode0_max(barcode0)
    #print(maxx, flush=True)
    filter_num = maxx*filter_ratio
    keep_edges = persistence_tree_edges(gtri, filtration_list, tri_edge_list, filter_num)


    adata.uns['persistence'] = {"ae_edges": ae_edges,
                                "filter_num": filter_num,

                                "delaunay_edges": tri_edge_list,
                                "filtration_list": filtration_list,
                                "simplextree_tri":simplex_tree,

                                "delaunay_edges_ae": tri_edge_list_ae,
                                "filtration_list_ae": filtration_list_ae,
                                "simplextree_tri_ae":simplex_tree_ae,
                                }




    ## 4. create triangulated graph
    adata.uns[f'{graph_name}_triangulation'] = reset_edges(gtri, keep_edges)
    adata.uns[f'{graph_name}_triangulation_circle'] = reset_edges(adata.uns[f'{graph_name}_triangulation'], ae_edges, keep_old=True)
    ## 5. calc_layout
    if calc_layout:
        print("calculating layout")
        pydot_layouts = nx.nx_pydot.graphviz_layout(adata.uns[f"{graph_name}_triangulation_circle"])
        adata.obsm[f'{graph_name}_triangulation_circle'] = np.array([pydot_layouts[i] for i in range(len(pydot_layouts))])

    return adata if iscopy else None
#endf construct_delaunay_persistence




def construct_delaunay(adata:AnnData,
                       graph_name:str=None,
                       layout_name:str=None,
                       trunc_quantile:float=0.75,
                       trunc_times:float=3,
                       cluster_name:str='group',
                       circle_quant=0.1,
                       node_attr='u',
                       start_n=5,
                       end_n = 5,
                       separate_ends_triangle = True,
                       random_seed = 2022,
                       calc_layout:bool = False,
                       iscopy:bool=False,
                       ):
    """
    Main function for constructing delaunay graph from a layout to get holes
     1. triangulate using the layout given
     2. remove too long distance edges by trunc_quantile * trunc_times
     3. randomly select starts and ends points by node_attr score
     4. connect starts and ends points with delaunay triangulation
     5. run layout algorithm for graph with holes  if calc_layout is true


    Parameters
    ----------
    adata: AnnData
        AnnData object
    graph_name: str
        graph name in adata.uns
    layout_name: str
        layout name in adata.obsm
    trunc_quantile: float
        quantile for truncating delaunay edges
    trunc_times: float
        times for truncating delaunay edges
    cluster_name: str
        cluster name in adata.obs
    circle_quant: float
        quantile for connecting start and end points
    node_attr: str
        node attribute for connecting
    start_n: int
        number of start points
    end_n: int
        number of end points
    separate_ends_triangle: bool
        whether to separate end points to different clusters
    random_seed: int
        random seed for selecting start and end points
    calc_layout: bool
        whether to calculate layout for the delaunay graph
    iscopy: bool
        whether to return a copy of adata
    """

    adata = adata.copy if iscopy else adata
    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")

    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")

    edges = truncated_delaunay(adata.uns[graph_name].nodes,
                               adata.obsm[layout_name],
                               trunc_quantile=trunc_quantile,
                               trunc_times=trunc_times)
    adata.uns[f'{graph_name}_triangulation'] = reset_edges(adata.uns[graph_name], edges, keep_old=False)
    construct_circle_delaunay(adata,
                              graph_name=f'{graph_name}_triangulation',
                              layout_name=layout_name,
                              cluster_name=cluster_name,
                              quant=circle_quant,
                              node_attr=node_attr,
                              start_n=start_n,
                              end_n = end_n,
                              separate_ends_triangle = separate_ends_triangle,
                              random_seed = random_seed,
                              calc_layout = calc_layout,
                              iscopy=False)
    return adata if iscopy else None
#end construct_delaunay

def construct_trucated_delaunay(adata:AnnData,
                                graph_name:str=None,
                                layout_name:str=None,
                                trunc_quantile:float=0.75,
                                trunc_times:float=3,
                                iscopy:bool=False,
                                ):
    """
    Function for only constructing delaunay graph from a layout without creating holes
     1. triangulate using the layout given
     2. remove too long distance edges by trunc_quantile * trunc_times

    Parameters
    ----------
    adata: AnnData
        AnnData object
    graph_name: str
        graph name in adata.uns
    layout_name: str
        layout name in adata.obsm
    trunc_quantile: float
        quantile for truncating delaunay edges
    trunc_times: float
        times for truncating delaunay edges
    is_copy: bool
        whether to return a copy of adata
    """
    if iscopy:
        adata = adata.copy()

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")

    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")


    edges = truncated_delaunay(adata.uns[graph_name].nodes,  adata.obsm[layout_name], trunc_quantile=trunc_quantile, trunc_times=trunc_times)
    adata.uns[f'{graph_name}_triangulation'] = reset_edges(adata.uns[graph_name], edges, keep_old=False)

    return adata if iscopy else None
#end construct_trucated_delaunay

def construct_circle_delaunay(adata:AnnData,
                              graph_name:str=None,
                              layout_name:str=None,
                              cluster_name:str='group',
                              quant=0.1,
                              node_attr='u',
                              start_n=5,
                              end_n = 5,
                              separate_ends_triangle = True,
                              random_seed = 2022,
                              calc_layout:bool = False,
                              iscopy:bool=False,
                              ):
    """
    Fuction for only connecting start and end points with delaunay triangulation
     3. randomly select starts and ends points by node_attr score
     4. connect starts and ends points with delaunay triangulation
     5. run layout algorithm for graph with holes  if calc_layout is true

    Parameters
    ----------
    adata: AnnData
        AnnData object
    graph_name: str
        graph name in adata.uns
    layout_name: str
        layout name in adata.obsm
    cluster_name: str
        cluster name in adata.obs
    quant: float
        quantile for connecting start and end points
    node_attr: str
        node attribute for connecting
    start_n: int
        number of start points
    end_n: int
        number of end points
    separate_ends_triangle: bool
        whether to separate end points to different clusters
    random_seed: int
        random seed for selecting start and end points
    calc_layout: bool
        whether to calculate layout for the delaunay graph
    iscopy: bool
        whether to return a copy of adata

    """
    if iscopy:
        adata = adata.copy()

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation"

    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

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


def truncated_delaunay(nodes, position, trunc_quantile=0.75, trunc_times=3):
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
    while True: ## only connected graph approved
        threshold = np.quantile(edges_distance, trunc_quantile) * trunc_times
        keep_edges = [tri_edges[i] for i in range(len(tri_edges)) if edges_distance[i] < threshold]
        tmpG = nx.Graph()
        tmpG.add_nodes_from(nodes)
        tmpG.add_edges_from(keep_edges)
        #print("is connected?: ", nx.is_connected(tmpG), trunc_quantile, trunc_times)
        if nx.is_connected(tmpG):
            break
        else:
            trunc_quantile += 0.01
            #threshold = np.quantile(edges_distance, trunc_quantile) * trunc_times

        if trunc_quantile >= 1:
            print("full delaunay connected graph")
            threshold = np.quantile(edges_distance, 0.999) * trunc_times
            keep_edges = [tri_edges[i] for i in range(len(tri_edges)) if edges_distance[i] < threshold]
            break


    return keep_edges


def construct_trucated_delaunay_knn(adata:AnnData,
                           graph_name:str=None,
                           layout_name:str=None,
                           A= None,
                           W= None,
                           knn_edges_k = 40,
                           iscopy:bool=False,
):
    """
    delaunay on a layout
    use diffusion knn edges to remove edges

    Parameters
    ---------
    position: 2D pd.array for the running of delaunay
    """
    if iscopy:
        adata = adata.copy()

    if "graph_basis" in adata.uns and not graph_name:
        graph_name = adata.uns["graph_basis"]
        A = re.sub("_g$", "_A", "{graph_name}")
        W = re.sub("_g$", "_W", "{graph_name}")
    if "graph_basis" in adata.uns and not layout_name:
        layout_name = adata.uns["graph_basis"]

    if graph_name not in adata.uns:
        raise ValueError(f"{graph_name} not in adata.uns")

    if layout_name not in adata.obsm:
        raise ValueError(f"{layout_name} not in adata.obsm")

    position = adata.obsm[layout_name]
    if type(position) == dict:
        position = np.array([position[x] for x in range(max(position.keys()) + 1)])
    tri = Delaunay(position)
    ti = tuple_increase
    tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
    tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten

    tmpG = reset_edges(adata.uns[graph_name], tri_edges, keep_old=False)
    while not nx.is_connected(tmpG):
        print("warn: not able to delaunay _triangulate, add gussain noisy offset!")
        noise_sigma_ratio=0.01

        rg1 = np.max(position[:, 0]) - np.min(position[:, 0])
        rg2 = np.max(position[:, 1]) - np.min(position[:, 1])
        m1 = np.mean(position[:, 0])
        m2 = np.mean(position[:, 1])
        X_noise=np.random.normal(m1, noise_sigma_ratio*rg1, size=position.shape[0])
        Y_noise=np.random.normal(m2, noise_sigma_ratio*rg2, size=position.shape[0])
        position[:, 0] += X_noise
        position[:, 1] += Y_noise
        tri = Delaunay(position)
        ti = tuple_increase
        tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
        tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
        tmpG = reset_edges(adata.uns[graph_name], tri_edges, keep_old=False)


    knn_edges = adjedges(adata.uns[A], adata.uns[W], knn_edges_k)
    knn_edges = set([tuple_increase(i,j) for (i,j) in knn_edges])
    #knn_edges |= set([tuple_increase(i,j)[::-1] for (i,j) in knn_edges])
    keep_edges = [i for i in tri_edges if i in knn_edges]
    while True:
        tmpG = reset_edges(adata.uns[graph_name], keep_edges, keep_old=False)
        if nx.is_connected(tmpG):
            adata.uns[f'{graph_name}_triangulation'] = tmpG
            break
        knn_edges_k *= 2
        knn_edges = adjedges(adata.uns[A], adata.uns[W], knn_edges_k)
        knn_edges = set([tuple_increase(i,j) for (i,j) in knn_edges])
        print("knn_edges_k:", knn_edges_k, len(knn_edges))
        keep_edges = [i for i in tri_edges if i in knn_edges]


    print("knn_k applied:", knn_edges_k)
    return adata if iscopy else None
#endf triangulation_delaunay_knn



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
                                      separate_ends_triangle = True,
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


    values = np.fromiter(nx.get_node_attributes(g, node_attr).values(), dtype=np.float32)

    n=len(g.nodes())
    o=np.sort(values)
    early=np.where(values<=o[round(n*quant)])[0]
    later=np.where(values>=o[min(round(n*(1-quant)), n-1)])[0]

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
    #print('u', u)



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
        #print(len(tri_edges))
        tri_edges = [(x,y) for (x,y) in tri_edges if not is_in_2sets(x,y, end_nodes_sets)]
        #print(len(tri_edges))
        ## filtering, ends should not be together.
        G_ae.add_edges_from(tri_edges)

    return G_ae
#endf connect_starts_ends_with_Delaunay

def connect_starts_ends_with_Delaunay_edges(g,
                                            layouts,
                                            group,
                                            quant=0.1,
                                            node_attr='u',
                                            start_n=5,
                                            end_n = 5,
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
    random_seed:

    Return
    -----
    edges: The starts ends edges exclude the same cluster edges
    """
    random.seed(random_seed)


    values = np.fromiter(nx.get_node_attributes(g, node_attr).values(), dtype=np.float32)

    n=len(g.nodes())
    o=np.sort(values)
    early=np.where(values<=o[round(n*quant)])[0]
    later=np.where(values>=o[min(round(n*(1-quant)), n-1)])[0]

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

    starts = early
    ends = later
    u = nx.get_node_attributes(g, node_attr)
    #print('u', u)

    #print("starts:", starts)
    #print("ends:", ends)
    #print("len(group): ", len(group))


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
    n_end_nodes_arr = [top_n_from(arr, u, min(end_n, len(arr)), largest=True) for arr in end_nodes_arr]
    n_end_nodes= [y for x in end_nodes_arr for y in x]

    ti = tuple_increase
    tri_edges_list = []
    for i in range(len(n_end_nodes_arr)):
        selected_nodes = list(n_start_nodes) + list(n_end_nodes_arr[i])
        tri = Delaunay(layouts[selected_nodes])
        tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
        tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
        tri_edges = [(selected_nodes[x], selected_nodes[y]) for (x,y) in tri_edges]
        tri_edges = [(x,y) for (x,y) in tri_edges if group[x] != group[y]]
        tri_edges_list.extend(tri_edges)

    return tri_edges_list
#endf connect_starts_ends_with_Delaunay_edges


def connect_starts_ends_with_Delaunay_customize(g,
                                      layouts,
                                      group=['0', '3', '5', '7'],
                                      start_clusters = ['0'],
                                      end_clusters = ['3', '5', '7'],
                                      node_attr='u',
                                      start_n=10,
                                      end_n = 10,
                                      ):
    """
    connect starts and the ends with curated start and end clusters.

    Parameters
    -------
    g: the graph generated by delaunay
    group: groups of each cell
    start_clusters: start clusters for creating holes
    end_clusters: end clusters
    start_n:
    end_n:

    Return
    -----
    G_ae: connected Delaunay graph for L1 holes
    """
    from scipy.spatial import Delaunay,distance
    import copy

    if len(start_clusters) <=0 or len(end_clusters) <=0:
        raise ValueError("start_clusters and end_clusters must be non-empty")

    for i in start_clusters:
        if i not in set(group):
            raise ValueError("start_clusters must be in group")
    for i in end_clusters:
        if i not in set(group):
            raise ValueError("end_clusters must be in group")

    if start_n <=0 or end_n <=0:
        raise ValueError("start_n and end_n must be positive")
    ## might be networkx.Graph layouts dict, convert to list
    if type(layouts) == dict:
        layouts = np.array([layouts[x] for x in range(max(layouts.keys()) + 1)])
    if not isinstance(layouts, np.ndarray):
        layouts = np.array(layouts)

    G_ae = copy.deepcopy(g)
    #starts = early
    #ends = later
    u = nx.get_node_attributes(g, node_attr)
    #print(u)

    start_cts = start_clusters
    end_cts = list(set(end_clusters))
    print("start clusters ", start_cts)
    print("end clusters ", end_cts)
    start_nodes = np.concatenate([np.where(np.array(group) == start_ct)[0] for start_ct in start_cts]).ravel()

    #print(start_nodes)
    n_start_nodes = top_n_from(start_nodes, u, min(start_n, len(start_nodes)), largest=False)

    end_nodes_arr = []
    for i in range(len(end_cts)):
        end_nodes_arr.append([i for i in np.where(np.array(group) == end_cts[i])[0] ])

    end_nodes_sets = [set(x) for x in end_nodes_arr]
    #print(end_nodes_arr)
    #print(end_n)
    n_end_nodes_arr = [top_n_from(arr, u, min(end_n, len(arr)), largest=True) for arr in end_nodes_arr]
    n_end_nodes= [y for x in end_nodes_arr for y in x]

    ti = tuple_increase
    for i in range(len(n_end_nodes_arr)):
        selected_nodes = list(n_start_nodes) + list(n_end_nodes_arr[i])
        tri = Delaunay(layouts[selected_nodes])
        tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
        tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
        tri_edges = [(selected_nodes[x], selected_nodes[y]) for (x,y) in tri_edges]
        #print(len(tri_edges))
        G_ae.add_edges_from(tri_edges)

    return G_ae



def connect_starts_ends_with_Delaunay3d(g,
                                        layouts,
                                        group,
                                        quant=0.1,
                                        node_attr='u',
                                        start_n=5,
                                        end_n = 5,
                                        random_seed = 2022
                                        ):

    ti = tuple_increase
    values = np.fromiter(nx.get_node_attributes(g, node_attr).values(), dtype=np.float32)

    n=len(g.nodes())
    o=np.sort(values)
    early=np.where(values<=o[round(n*quant)])[0]
    later=np.where(values>=o[min(round(n*(1-quant)), n-1)])[0]

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
    #print('u', u)

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

    n_end_nodes_arr = [top_n_from(arr, u, min(end_n, len(arr)), largest=True) for arr in end_nodes_arr]
    n_end_nodes= [y for x in end_nodes_arr for y in x]

    for i in range(len(n_end_nodes_arr)):
        selected_nodes = list(n_start_nodes) + list(n_end_nodes_arr[i])
        tri = Delaunay(layouts[selected_nodes])
        tri_edges =[[ti(a,b),ti(a,c),ti(a,d),ti(b,c), ti(b,d), ti(c,d)] for a,b,c,d in tri.simplices]
        tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
        tri_edges = [(selected_nodes[x], selected_nodes[y]) for (x,y) in tri_edges]
        G_ae.add_edges_from(tri_edges)

    return G_ae
#endf connect_starts_ends_with_Delaunay3d


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

def simplextree(layouts):
    import gudhi as gd
    point_coordinates = layouts


    edge_list = []
    filtration_list = []

    # extract connecting edges <-- Something is not quite right here
    #for edge in graph.edges:
    #    if np.linalg.norm(point_coordinates[edge[0]]-point_coordinates[edge[1]])>500:
    #        edge_list.append(edge)
    #        filtration_list.append(0)

    #Add triangulation edges
    delaunay_triangulation = Delaunay(point_coordinates)
    simplices = delaunay_triangulation.simplices
    simplex_tree = gd.SimplexTree()

    ti = tuple_increase
    for simplex in simplices:
        for i, j in zip(simplex, np.roll(simplex, -1)):
            edge_list.append(ti(i,j))
            filtration_list.append(np.linalg.norm(point_coordinates[i] - point_coordinates[j]))

    #Construct simplicial complex from graph
    simplex_tree.insert_batch(np.array(edge_list).T, np.array(filtration_list))
    #add all possible triangles
    simplex_tree.expansion(2)

    return edge_list, filtration_list, simplex_tree



def simplextree_ae(graph, layouts):
    import gudhi as gd
    point_coordinates = layouts


    edge_list = []
    filtration_list = []

    # extract connecting edges <-- Something is not quite right here
    for edge in graph.edges:
        #if np.linalg.norm(point_coordinates[edge[0]]-point_coordinates[edge[1]])>500:
        edge_list.append(edge)
        filtration_list.append(0)

    #Add triangulation edges
    delaunay_triangulation = Delaunay(point_coordinates)
    simplices = delaunay_triangulation.simplices
    simplex_tree = gd.SimplexTree()

    ti = tuple_increase
    for simplex in simplices:
        for i, j in zip(simplex, np.roll(simplex, -1)):
            edge_list.append(ti(i,j))
            filtration_list.append(np.linalg.norm(point_coordinates[i] - point_coordinates[j]))

    #Construct simplicial complex from graph
    simplex_tree.insert_batch(np.array(edge_list).T, np.array(filtration_list))
    #add all possible triangles
    simplex_tree.expansion(2)

    return edge_list, filtration_list, simplex_tree


def barcode0_max(barcode0):
    barcode0 = [j for i, j in barcode0]
    barcode0 = [j for i in barcode0 for j in i]
    barcode0 = [i for i in barcode0 if i < 1e308]
    return max(barcode0)


def persistence_tree_edges(g, filtration_list, edge_list, filter_num):
    index = np.where(np.array(filtration_list) <= filter_num)[0]
    keep_edges = np.array(edge_list)[index]
    return keep_edges
