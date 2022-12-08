import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix
from .util import lexsort_rows

## import from https://git.rwth-aachen.de/netsci/trajectory-outlier-detection-flow-embeddings/
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




def create_normalized_l1(B1, B2, mode="RW"):
    if B2.shape[1] > 0:
        d1 = np.sum(np.abs(B2), axis=1)
        d1 = np.asarray(d1.flatten())[0]
        D1 = np.diag(np.maximum(1, d1))
        D1inv = np.diag(np.divide(1, np.diag(D1)))
    else:
        num_edges = B1.shape[1]
        D1 = np.eye(num_edges)
        D1inv = D1

    d0weighted = np.maximum(1, np.sum(np.abs(B1 * D1), axis=1))
    d0weighted = np.reshape(d0weighted, (d0weighted.size))
    D0weighted = np.diag(d0weighted)
    D0weightedinv = np.diag(np.divide(1, d0weighted))

    # assemble normalized Laplacian
    #L1 = D1*B1.T*1/2*D0weightedinv*B1 + B2*1/3*B2.T*D1inv
    L1_node = ((D1 * B1.T * (1/2)) @ (D0weightedinv)) * B1

    if B2.shape[1] > 0:
        L1_edge = B2 * 1/3 * B2.T * D1inv
    else:
        L1_edge = 0

    L1 = L1_node + L1_edge
    if mode == "sym":
        L1 = np.sqrt(D1inv) * L1 * np.sqrt(D1inv)
    return L1, D1, D1inv, D0weighted, D0weightedinv

def create_weighted_edge_triangle_incidence_matrix(G, elist, tlist):
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

        w1 = G[e1[0]][e1[1]]['weight']
        w2 = G[e2[0]][e2[1]]['weight']
        w3 = G[e3[0]][e3[1]]['weight']

        data.append(w1)
        row_ind.append(elist_dict[tuple(e1)])
        col_ind.append(i)

        data.append(w2)
        row_ind.append(elist_dict[tuple(e2)])
        col_ind.append(i)

        data.append(w3)
        row_ind.append(elist_dict[tuple(e3)])
        col_ind.append(i)

    B2 = csc_matrix((np.array(data), (np.array(row_ind), np.array(
        col_ind))), shape=(len(elist), len(tlist)), dtype=np.int8)
    return B2
