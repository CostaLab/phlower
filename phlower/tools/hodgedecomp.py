import time
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix
from typing import Union
from numpy.linalg import qr,solve,lstsq
from .incidence import *
from ..util import find_knee, test_cholesky
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def lexsort_rows(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return array[np.lexsort(np.rot90(array))]

def triangle_list(G: nx.Graph) -> np.ndarray:
    triangles = []
    for i, j in G.edges():
        if i > j:
            i, j = j, i

        # We filter the neighbors to be larger than the edge's incident nodes.
        # This way we guarantee uniqueness of the tuples we find.
        first_node_neighbors = set(filter(lambda t: t > j, G[i]))
        second_node_neighbors = set(filter(lambda t: t > j, G[j]))

        # find intersection between those neighbors => triangle
        common_neighbors = first_node_neighbors & second_node_neighbors

        for t in common_neighbors:
            assert i < j < t
            triangles.append([i, j, t])

    result = np.array(triangles)
    return lexsort_rows(result)


#def gradop(g:nx.DiGraph) -> csc_matrix : ## construct B1.T matrix, node to edge matrix
def gradop(g:nx.DiGraph): ## construct B1.T matrix, node to edge matrix
  return nx.incidence_matrix(g, oriented=True).T


def div_adj(adj_matrix: Union[np.ndarray, csr_matrix, csr_matrix], tol:float=1e-7) -> csr_matrix:
    """
    divergence directly from adjacency matrix
    """
    if isinstance(adj_matrix, csr_matrix):
        adj_matrix = adj_matrix.toarray()
        adj_matrix[adj_matrix < tol] = 0
    else:
        adj_matrix.data[adj_matrix.data < tol] = 0
        adj_matrix.eliminate_zeros()

    edge = np.array(adj_matrix.nonzero())
    ne = edge.shape[1]
    nv = adj_matrix.shape[0]
    e_w = adj_matrix[edge[0, :], edge[1, :]].T
    #print("ew_shape", e_w.shape)
    i, j, x = np.tile(range(ne), 2), edge.flatten(), np.repeat([-1, 1], ne)
    #print("divop shape", scipy.sparse.csr_matrix((x, (i, j)), shape=(ne, nv)).T.shape)
    return -1 * scipy.sparse.csr_matrix((x, (i, j)), shape=(ne, nv)).T @ e_w


#-> csc_matrix
def divop(g:nx.DiGraph) :
    return -1 * gradop(g).T


def curlop(g):

    elist = [(x[0], x[1]) for x in g.edges()]
    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}
    tlist = triangle_list(g)
    data = []
    row_ind = []
    col_ind = []
    for i, t in enumerate(tlist):
        e1 = t[[0, 1]]
        e2 = t[[1, 2]]
        e3 = t[[0, 2]]

        data.append(1)
        row_ind.append(elist_dict[tuple(e1)])
        col_ind.append(i)

        data.append(1)
        row_ind.append(elist_dict[tuple(e2)])
        col_ind.append(i)

        data.append(-1)
        row_ind.append(elist_dict[tuple(e3)])
        col_ind.append(i)

    B2 = csc_matrix((np.array(data), (np.array(row_ind), np.array(
        col_ind))), shape=(len(elist), len(tlist)), dtype=np.int8)
    return B2.T


def laplacian0(g:nx.DiGraph) -> csc_matrix:
    gradop_g = gradop(g)
    return gradop_g.T@gradop_g

def laplacian1(g:nx.DiGraph) -> csc_matrix:
    gradop_g = gradop(g)
    curlop_g = curlop(g)
    return gradop_g@gradop_g.T + curlop_g.T@curlop_g



def potential(g:nx.DiGraph, tol=1e-7, weight_attr='weight', method='lstsq'):
    L = nx.laplacian_matrix(g.to_undirected(), weight=None)
    #1st
    #p = solve(L.toarray(), -div(g))

    #2nd implementation: slower, don't know why ddhodge use this.
    #Q, R = qr(L.toarray())    # QR decomposition with qr function
    #y = np.dot(Q.T, -div(g))  # Let y=Q'.B using matrix multiplication
    #p = np.linalg.solve(R, y)

    # 3rd implementation
    #p =  lstsq(L.toarray(), -div(g, weight_attr), rcond=None)[0]

    #4th
    if method == 'lstsq':
        p =  lstsq(L.toarray(), -div(g, weight_attr), rcond=None)[0]
    elif method == 'lsmr':
        p = scipy.sparse.linalg.lsmr(L, -div(g, weight_attr))[0]
    elif method == 'lsqr':
        p = scipy.sparse.linalg.lsqr(L, -div(g, weight_attr))[0]
    elif method == 'cholesky':
        #p =
        ret = test_cholesky(L, verbose=True)
        if ret:
            p = ret(-div(g, weight_attr))
        else:
            p = scipy.sparse.linalg.spsolve(L, -div(g))
    else:
        raise ValueError("method must be one of 'lstsq', 'lsmr', 'lsqr', 'cholesky'")

    return (p - min(p))


def grad(g:nx.DiGraph, tol=1e-7, weight_attr='weight', lstsq_method='lstsq'):
    return gradop(g)@ potential(g, tol, weight_attr=weight_attr, method=lstsq_method)


def div(g:nx.DiGraph, weight_attr='weight'):
    return divop(g) @ np.fromiter(nx.get_edge_attributes(g, weight_attr).values(), dtype=float) ##TODO please check the order of weights
    #return divop(g) @ nx.to_scipy_sparse_matrix(g, weight=weight_attr)

def curl(g, weight_attr="weight"):
    return curlop(g) @ np.fromiter(nx.get_edge_attributes(g, weight_attr).values(), dtype=float)
    #return curlop(g) @ nx.to_scipy_sparse_matrix(g, weight=weight_attr)




def L1Norm_decomp(adata: AnnData,
                  graph_name: str = None,
                  eigen_num: int = 100,
                  L1_mode = "sym",
                  mysym = 'a',
                  check_symmetric: bool = True,
                  isnorm = True,
                  iscopy: bool = False,
        ):

    """
    graph hodge laplacian decomposition
    if sym:
        call eigsh to for symmetric matrix eigen decomposition

        L_1^s = D_2^{-1/2} L_1_norm D_2^{1/2}
        eigen vector of which is D_2^{-1/2} u_r
    else:
        call eigs to for non-symmetric matrix eigen decomposition, and only keep the real part of eigen values and vectors
        L1_norm = D_2 B_1^\top D_1^{-1} B_1 + B_2 D_3 B_2^\top D_2^{-1}

    Parameters
    ----------
    adata: AnnData
        AnnData object store the graph in uns slot
    graph_name: str
        graph name for the graph with many holes
    eigen_num: int
        number of eigenvalues to be calculated
    L1_mode: str
        "sym" or "RW"
    check_symmetric: bool
        check if the matrix is symmetric
    isnorm: bool
        normalize the graph hodge laplacian
    iscopy: bool
        copy the adata or not
    """
    if iscopy:
        adata = adata.copy()

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    if graph_name not in adata.uns.keys():
        raise Exception(f"graph {graph_name} not found in adata.uns")

    elist = np.array(adata.uns[graph_name].edges())
    tlist = triangle_list(adata.uns[graph_name])

    B1 = create_node_edge_incidence_matrix(elist)
    B2 = create_edge_triangle_incidence_matrix(elist, tlist)

    #L1all = create_normalized_l1(B1, B2, mode="RW")
    D2=None
    if isnorm:
        L1all = create_normalized_l1(B1, B2, mode=L1_mode)
        L1 = L1all[0]
        D2 = L1all[1]
        #if not scipy.linalg.issymmetric(L1):
            #L1 = np.tril(L1) + np.triu(L1.T, 1)
            #L1 = 1/2*(L1 + L1.T)
            #L1 =  np.maximum(L1, L1.T)
    else:
        L1all = create_l1(B1, B2)
        L1 = L1all[0]
        D2 = L1all[1]
        #L1 = L1all[0].toarray()

    ##TODO: for testing, remove this block after
    if mysym == 'a+at':
        L1 = 1/2*(L1 + L1.T)
    elif mysym == 'max':
        L1 = np.maximum(L1, L1.T)
    elif mysym == 'min':
        L1 = np.minimum(L1, L1.T)
    elif mysym == 'a':
        L1 = L1
    elif mysym == 'upper':
        L1 = np.triu(L1) + np.triu(L1.T, 1)
    elif mysym == 'lower':
        L1 = np.tril(L1) + np.tril(L1.T, -1)


    start = time.time()
    if L1_mode == "sym": #there's no need to check the symmetry of L1
        check_symmetric = False
    d = harmonic_projection_matrix_with_w(L1.astype(float), eigen_num, check_symmetric = check_symmetric)
    end = time.time()

    ## u_R = D_2^(1/2) * u
    ## u_L^\top = u^\top * D_1^(-1/2)
    print((end-start), " sec")


    #adata.uns[f'{graph_name}_L1Norm'] = L1all

    adata.uns[f'{graph_name}_L1Norm_decomp_vector'] = d['v'] if L1_mode == "RW" else d['v'] @ np.sqrt(D2)
    adata.uns[f'{graph_name}_L1Norm_decomp_value'] = d['w'] if L1_mode == "RW" else d['w']

    #adata.uns[f'{graph_name}_L1Norm_decomp_vector'] = d['v'] if L1_mode == "RW" else d['v']
    #adata.uns[f'{graph_name}_L1Norm_decomp_value'] = d['w'] if L1_mode == "RW" else d['w']




    #adata.uns['X_pca_ddhodge_g_triangulation_circle_B1'] = B1
    #adata.uns['X_pca_ddhodge_g_triangulation_circle_B2'] = B2
    return adata if iscopy else None

def knee_eigen(adata: AnnData,
               eigens: Union[str, np.ndarray] = None,
               plot = False,
               iscopy = False,
               ):

    if iscopy:
        adata = adata.copy()

    if "graph_basis" in adata.uns.keys() and not eigens:
        eigens = adata.uns["graph_basis"] + "_triangulation_circle_L1Norm_decomp_value"

    if isinstance(eigens, str):
        eigens = adata.uns[eigens]

    x = range(1, len(eigens) + 1)
    y = eigens
    idx = find_knee(x, y, plot = plot)
    adata.uns['eigen_value_knee'] = idx
    print("knee eigen value is ", idx)

    return adata if iscopy else None

