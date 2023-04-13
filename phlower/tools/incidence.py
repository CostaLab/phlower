## imported and adjusted from https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/
import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix, linalg, csr_matrix
import scipy
from ..util import lexsort_rows


def harmonic_projection_matrix_with_w(L1: csr_matrix, number_of_holes: int, check_symmetric:bool=True) -> dict:
    """
    Computes the harmonic projection matrix for the simplicial complex with the
    given Hodge-1 Laplacian.

    Parameters
    ----------
    L1 : csr_matrix of type float
    number_of_holes : int

    Return
    ----------
    dict
        w: eigen values
        v: eigen vectors rows
    """
    if (not check_symmetric) or scipy.linalg.issymmetric(L1.toarray(), atol=1e-08, rtol=1e-05):
        w, v = linalg.eigsh(L1, k=number_of_holes,
                        v0=np.ones(L1.shape[0]), which='SM')
    else:

        w, v = linalg.eigs(L1, k=number_of_holes,
                        v0=np.ones(L1.shape[0]), which='SM')
        idx = w.argsort()
        w = w[idx]
        w = np.real(w)
        v = v[:, idx]
        v = np.real(v)

    return {"w":w, "v":v.T}

def assign_eweight(G, A):
    """
    G: graph
    A: nodes x nodes matrix(u_i - u_j)
    """
    pass
    d_edge_attr = {(x,y):{"weight": A[x,y]}  for (x,y) in G.edges()}
    nx.set_edge_attributes(G, d_edge_attr)
    return G
#endf



#https://math.stackexchange.com/questions/1088441/the-incidence-matrix-of-a-weighted-graph
#Biyikoglu, T., Leydold, J., & Stadler, P. F. (2007). Laplacian eigenvectors of graphs: Perron-Frobenius and Faber-Krahn type theorems. Springer.
#Kelner, J. (2007). An Algorithmist’s Toolkit: Lecture Notes. Lecture 2. MIT.

def create_weighted_node_edge_incidence_matrix(G, attr='weight'):
    B1w = nx.incidence_matrix(G, weight='weight')
    return B1w



def create_node_edge_incidence_matrix(elist):
    num_edges = len(elist)
    data = [-1] * num_edges + [1] * num_edges
    row_ind = [e[0] for e in elist] + [e[1] for e in elist]
    col_ind = [i for i in range(len(elist))] * 2
    B1 = csc_matrix(
        (np.array(data), (np.array(row_ind), np.array(col_ind))), dtype=np.int8)
    return B1


def create_edge_triangle_incidence_matrix(elist, tlist):
    if len(tlist) == 0:
        return csc_matrix([], shape=(len(elist), 0), dtype=np.int8)

    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}

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
    return B2


def create_l1(B1, B2):
    L1 = (B1.T @ B1 + B2 @ B2.T)

    if B2.shape[1] > 0:
        d2 = np.sum(np.abs(B2), axis=1)
        d2 = np.asarray(d2.flatten())[0]
        D2 = scipy.sparse.diags(np.maximum(1, d2), 0, format="csr")
    else:
        num_edges = B1.shape[1]
        D2 = scipy.sparse.eye(num_edges)
    #L1 = B1.T.dot(B1) + B2.dot(B2.T)
    return L1, D2



def create_normalized_l1(B1, B2, mode="sym"):
    if B2.shape[1] > 0:
        d2 = np.sum(np.abs(B2), axis=1)
        d2 = np.asarray(d2.flatten())[0]
        D2 = scipy.sparse.diags(np.maximum(1, d2), 0, format="csr")
        D2inv = scipy.sparse.diags(np.divide(1, D2.diagonal()), 0, format="csr")
    else:
        num_edges = B1.shape[1]
        D2 = scipy.sparse.eye(num_edges)
        D2inv = D2

    d1weighted = np.maximum(1, np.sum(np.abs(B1 @ D2), axis=1))
    d1weighted = np.reshape(d1weighted, (d1weighted.size))
    d1weighted = np.array(d1weighted).flatten()
    D1weighted = scipy.sparse.diags(d1weighted, 0, format="csr")
    D1weightedinv = scipy.sparse.diags(np.divide(1, d1weighted), 0, format="csr")

    # assemble normalized Laplacian
    #L1 = D2*B1.T*1/2*D1weightedinv*B1 + B2*1/3*B2.T*D2inv
    L1_node = ((D2 @ B1.T * (1/2)) @ (D1weightedinv)) @ B1

    if B2.shape[1] > 0:
        L1_edge = B2 * 1/3 @ B2.T @ D2inv
    else:
        L1_edge = 0

    L1 = L1_node + L1_edge
    if mode == "sym":
        #L1 = np.sqrt(D2inv) * L1 * np.sqrt(D2inv) ## from the gitlab: https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings
        ## L_1^S = D_2^{-1/2} L_1 D_2^{1/2}
        ## u_R = D_2^{1/2} u_1
        ## u_L^T = u_1^T D_2^{-1/2}
        L1 = np.sqrt(D2inv) * L1 * np.sqrt(D2)
    return L1, D2


def create_weighted_edge_triangle_incidence_matrix(G, elist, tlist, weight_attr="weight"):
    if len(tlist) == 0:
        return csc_matrix([], shape=(len(elist), 0), dtype=np.int8)

    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}

    data = []
    row_ind = []
    col_ind = []
    for i, t in enumerate(tlist):
        e1 = t[[0, 1]]
        e2 = t[[1, 2]]
        e3 = t[[0, 2]]

        w1 = G[e1[0]][e1[1]][weight_attr]
        w2 = G[e2[0]][e2[1]][weight_attr]
        w3 = G[e3[0]][e3[1]][weight_attr]

        data.append(w1)
        row_ind.append(elist_dict[tuple(e1)])
        col_ind.append(i)

        data.append(w2)
        row_ind.append(elist_dict[tuple(e2)])
        col_ind.append(i)

        #data.append(w3*-1)
        data.append(w3)
        row_ind.append(elist_dict[tuple(e3)])
        col_ind.append(i)

    B2w = csc_matrix((np.array(data), (np.array(row_ind), np.array(
        col_ind))), shape=(len(elist), len(tlist)), dtype=np.int8)
    return B2w


