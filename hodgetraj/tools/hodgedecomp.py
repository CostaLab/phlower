import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from numpy.linalg import qr,solve,lstsq


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


def gradop(g:nx.DiGraph) -> csc_matrix : ## construct B1.T matrix, node to edge matrix
  return nx.incidence_matrix(g, oriented=True).T


def divop(g:nx.DiGraph) -> csc_matrix:
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



def potential(g:nx.DiGraph, tol=1e-7, weight_attr='weight'):
    L = nx.laplacian_matrix(g.to_undirected(), weight=None)
    #1st
    #p = solve(L.toarray(), -div(g))

    #2nd implementation: slower, don't know why ddhodge use this.
    #Q, R = qr(L.toarray())    # QR decomposition with qr function
    #y = np.dot(Q.T, -div(g))  # Let y=Q'.B using matrix multiplication
    #p = np.linalg.solve(R, y)

    # 3rd implementation
    p =  lstsq(L.toarray(), -div(g, weight_attr), rcond=None)[0]

    return (p - min(p))


def grad(g:nx.DiGraph, tol=1e-7, weight_attr='weight'):
    return gradop(g)@ potential(g, tol, weight_attr=weight_attr)


def div(g:nx.DiGraph, weight_attr='weight'):
    return divop(g) @ np.fromiter(nx.get_edge_attributes(g, weight_attr).values(), dtype=float) ##TODO please check the order of weights

def curl(g, weight_attr="weight"):
    return curlop(g) @ np.fromiter(nx.get_edge_attributes(g, weight_attr).values(), dtype=float)


