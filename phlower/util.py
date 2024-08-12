import math
import scipy
import numpy as np
import itertools
import pandas as pd
import scipy.spatial
from collections import OrderedDict, Counter
from scipy.stats import gaussian_kde



def lexsort_rows(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return array[np.lexsort(np.rot90(array))]

def tuple_increase(a,b):
    if a < b:
        return (a,b)
    return(b,a)
#endf tuple_increase

def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array"""

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:,None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:,None], 2), axis=1))

    rho = upper / lower

    return rho


def norm01(a):
    maxx= max(a)
    minn = min(a)
    a = [maxx if np.isnan(i) else i for i in a]
    a = [(i - minn)/(maxx-minn) for i in a]
    return a
#endf norm01

def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
#endf pairwise

def top_n_from(node_arr, score_dict, n, largest=True):
    """
    Select top n largest or smallest nodes from node_arr according the score_dict

    Parameters
    -------
    node_arr: nodes candidates
    score_dict: score dictionary for all nodes
    n: top n
    largest: largest n if True else smallest
    """
    assert(n > 0)
    if isinstance(node_arr, list):
        node_arr = np.array(node_arr)
    score_arr = np.array([*score_dict.values()])[node_arr]
    if largest:
        node_idx = node_arr[np.argpartition(score_arr, -n)[-n:]]
        return node_idx
    else:
        node_idx = node_arr[np.argpartition(score_arr, n)[:n]]
        return node_idx

#endf top_n_from

def is_in_2sets(a,b, set_list):
    """
    if a, b pair in different set_list return True
    else return False
    """
    idx_as = [i for i in range(len(set_list)) if a in set_list[i]]
    idx_bs = [i for i in range(len(set_list)) if b in set_list[i]]
    if (not idx_as) or (not idx_bs):
        return False
    idx_a = idx_as[0]
    idx_b = idx_bs[0]

    if idx_a == idx_b:
        return False
    return True
#endf is_in_2sets

def cosine_dic(dic1,dic2):
    numerator = 0
    dena = 0
    for k1,v1 in dic1.items():
        numerator += v1*dic2.get(k1,0.0)
        dena += v1*v1
    denb = 0
    for v2 in dic2.values():
        denb += v2*v2
    return numerator/math.sqrt(dena*denb)

def kde_eastimate(trajectories, layouts, sample_n=4000, seeds=2022):
    """
    run gaussian_kde on trajectory nodes

    Parameters
    -----------
    trajectories: trajectories list
    layouts: x,y coordinate of each point
    sample_n: sample to reduce the computing time
    seeds: seeds for sampling
    """

    xy = [layouts[a] for t in trajectories for a in t]
    x = np.array([x for x, y in xy])
    y = np.array([y for x, y in xy])

    # Calculate the point density
    xy = np.vstack([x,y])

    np.random.seed(seeds)
    sub_idx = np.random.choice(range(xy.shape[1]), min(sample_n, len(x)), replace=False)
    xy = xy[:, sub_idx]
    x = x[sub_idx]
    y = y[sub_idx]
    z = gaussian_kde(xy)(xy)
    index = np.array([a for t in trajectories for a in t])[sub_idx]

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()

    x, y, z = x[idx], y[idx], z[idx]
    index = index[idx]
    unique_index = [i for i in OrderedDict((x, index.tolist().index(x)) for x in index).values()]

    return {'idx':index[unique_index], 'x':x[unique_index], 'y':y[unique_index], 'z':z[unique_index]}
#endf

def intersect_kde(traj_dicts, ratio=0.8):
    """
    use the density information to get the insert branching points

    Parameters
    ----------
    traj_dicts: trajectory density dictionary, key is each trajectory class, value is the density(x,y,z,idx)
    ratio: ratio to keep for the intersection

    Return
    ----------
    intersection of all nodes.
    """
    df_list = [(pd.DataFrame(traj)).drop_duplicates() for traj in traj_dicts.values()]
    set_list = [set((df.nlargest(int(len(df) *ratio), 'z'))['idx']) for df in df_list]
    return set.intersection(*set_list)
#endf


def get_uniform_multiplication(number):
    assert(number>0)
    sqrt_n = int(np.sqrt(number))
    if np.power(sqrt_n, 2) == number:
        return (sqrt_n, sqrt_n)
    elif np.multiply(sqrt_n, sqrt_n+1) > number:
        return (sqrt_n+1, sqrt_n)
    else:
        return (sqrt_n+1, sqrt_n+1)
#endf


def find_knee(x,y, plot=False):
    """
    find the knee point of the curve
    """
    assert(len(x) == len(y))

    """
    #deprecated, use distance to begin end segment, working not well
    import numpy.matlib
    allcoord = np.vstack((x, y)).T
    npoints = len(x)
    firstpoint = allcoord[0]
    line_vec = allcoord[-1] - allcoord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = allcoord - firstpoint
    scalar_product = np.sum(vec_from_first * np.matlib.repmat(line_vec_norm, npoints, 1), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    distToLine = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    idx_of_best_point = np.argmax(distToLine)
    """

    ev = y
    d = {}
    for idx, i in enumerate(ev):
        if idx ==0:
            continue
        d[idx] = round(np.abs(i/ev[idx-1]), 2)

    idx_of_best_point = max(d, key=d.get)

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        sns.lineplot(x=x, y=y, ax=ax, sort=False)
        ax.scatter(x=x[idx_of_best_point-1], y=y[idx_of_best_point-1], color='red')

    return idx_of_best_point



def networkx_node_to_df(G):
    """
    https://stackoverflow.com/questions/62383699/converting-networkx-graph-to-data-frame-with-its-attributes
    """
    nodes = {}
    for node, attribute in G.nodes(data=True):
        if not nodes.get('node'):
            nodes['node'] = [node]
        else:
            nodes['node'].append(node)

        for key, value in attribute.items():
            if not nodes.get(key):
                nodes[key] = [value]
            else:
                nodes[key].append(value)

    return pd.DataFrame(nodes)

def networkx_edge_to_df(G):
    edges = {}
    for source, target, attribute in G.edges(data=True):

        if not edges.get('source'):
            edges['source'] = [source]
        else:
            edges['source'].append(source)

        if not edges.get('target'):
            edges['target'] = [target]
        else:
            edges['target'].append(target)

        for key, value in attribute.items():
            if not edges.get(key):
                edges[key] = [value]
            else:
                edges[key].append(value)
    return pd.DataFrame(edges)

def is_node_attr_existing(G, attr):
    for node in G.nodes:
        node_dict = G.nodes[node]
        if attr in node_dict:
            return True
    return False

def is_edge_attr_existing(G, attr):
    for edge in G.edges:
        edge_dict = G.edges[edge]
        if attr in edge_dict:
            return True
    return False




def networkx_node_df_to_ebunch(df, nodename='node'):

    attributes = [col for col in df.columns if not col==nodename]

    ebunch = []

    for ix, row in df.iterrows():
        ebunch.append((row[nodename], {attribute:row[attribute] for attribute in attributes}))

    return ebunch

## copy from networkx to workaround lower version of networkx
def bfs_layers(G, sources):
    """Returns an iterator of all the layers in breadth-first search traversal.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Specify starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    layer: list of nodes
        Yields list of nodes at the same distance from sources
    """
    if sources in G:
        sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = list()
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer



def bsplit(facs, keep_empty=False):
    #facs = ['one','two','three']
    l1 = []
    l2 = []
    for pattern in itertools.product([True,False],repeat=len(facs)):
        l1.append([x[1] for x in zip(pattern,facs) if x[0]])
        l2.append([x[1] for x in zip(pattern,facs) if not x[0]])

    return [(tuple(set(l1[i])),tuple(set(l2[i]))) for i in range(len(l1)) if keep_empty or (l1[i] and l2[i])]


def term_frequency_cosine(list1, list2):
    """
    calculate the cosine similarity of two lists
    """
    c1 = Counter(list1)
    c2 = Counter(list2)
    terms = set(c1).union(c2)
    vec1 = np.array([c1.get(k, 0) for k in terms])
    vec2 = np.array([c2.get(k, 0) for k in terms])
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_cut_point(list_with_nan, cut_threshold=0.1, increase=False):
    """
    find the cut point of a list with nan
    when encounter value < cut_threshold, return the former index
    """
    compare = np.greater if increase else np.less

    length = len(list_with_nan)
    curr_candidate = 0
    for i in range(1, length):
        if np.isnan(list_with_nan[i]):
            continue
        if compare(list_with_nan[i], cut_threshold):
            break
        if not compare(list_with_nan[i], cut_threshold):
            curr_candidate = i
    return curr_candidate


def find_cut_point_bu(list_with_nan, cut_threshold=0.1, increase=False):
    """
    bottom up find the cut point of a list with nan
    when encounter value < cut_threshold, return the former index
    """
    compare = np.greater if increase else np.less


    length = len(list_with_nan)
    candidate = length - 1
    for i in range(length - 2, -1, -1):
        if np.isnan(list_with_nan[i]):
            continue
        if compare(list_with_nan[i], cut_threshold):
            candidate = i
            break
    if candidate == 0:
        return 0

    curr_candidate = candidate - 1
    for i in range(candidate, -1, -1):
        if np.isnan(list_with_nan[i]):
            continue
        if compare(list_with_nan[i], cut_threshold):
            continue

        if not compare(list_with_nan[i], cut_threshold):
            curr_candidate = i
            break

    return curr_candidate  if curr_candidate > 0 else 0

def test_cholesky( A, beta=1e-6, verbose=False ):
    """ try cholesky( a scipy.sparse matrix  + beta I )

    https://scicomp.stackexchange.com/questions/12979/testing-if-a-matrix-is-positive-semi-definite
    Why A + beta I, beta say 1e-6 ?
    If A has tiny eigenvalues, 0 to within machine precision,
    about half of these "zeros" may be negative -- tough on solvers.
    Also the condition number improves to ~ rho(A) / beta.
    """
    #scikit-sparse
    from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError, CholmodTooLargeError
    if not scipy.sparse.isspmatrix_csc(A):
        A = scipy.sparse.csc_matrix(A)
    try:
        solve = cholesky( A, beta=beta )  # A + beta I
        if verbose:
            print( "+ %g I is positive-definite" % (beta ))
        return solve  # x = solve( b )
    except CholmodNotPositiveDefiniteError:
        if verbose:
            print( " + %g I is not positive-definite" % (beta ))
    except CholmodTooLargeError:
        if verbose:
            print( " + %g I is too large" % (beta ))
        return False

def has_islands(adjacency_matrix):
    num_vertices = adjacency_matrix.shape[0]
    visited = np.zeros(num_vertices, dtype=bool)

    for vertex in range(num_vertices):
        if not visited[vertex]:
            stack = [vertex]
            visited[vertex] = True

            while stack:
                curr_vertex = stack.pop()

                neighbors = np.nonzero(adjacency_matrix[curr_vertex])[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)
                        visited[neighbor] = True

    return np.any(~visited)



def get_quantiles(v = None, length = None):
  if length is None:
    length = len(v)

  if len(v) < length:
    v2 = [0] * length
    v2[:len(v)] <- v[:len(v)]
  else:
    v2 = v
  order = v2.argsort()
  ranks = order.argsort()

  p = np.trunc(ranks)/len(v2)
  if len(v) < length:
    p <- p[:len(v)]

  return p


def module_check_install(module_name):
    import os
    try:
        __import__(module_name)
    except ImportError:
        print("Installing %s" % module_name)
        os.system (f"pip install {module_name}")



def express_prop(adata, gene, obs_col='time_int'):
    from collections import Counter
    d = Counter(adata.obs[obs_col])

    is_expressed = [int(i) for i in adata[:, gene].X.toarray().ravel() > 0]
    df = pd.DataFrame({"is_expressed":is_expressed, "days": adata.obs[obs_col]})
    mdf = df.groupby('days').sum()
    mdf
    mdf['total'] = [d[k] for k in mdf.index]
    mdf['expressed_prop'] = [round(a/b,2) for a,b in zip(mdf['is_expressed'], mdf['total'])]

    return mdf

def custom_cmap(color_range=['grey', 'red'], name='mycmap'):
    """
    custome gradient colormap using colorir
    """
    import colorir as cir
    import matplotlib as mpl
    color_range = [i if i.startswith('#') else mpl.colors.cnames[i] for i in color_range  ]
    grad = cir.PolarGrad(color_range)#.to_cmap()
    cmap_ = grad.to_cmap(name, 256)
    return cmap_
