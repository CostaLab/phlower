import sklearn
import copy
import igraph
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain
from typing import Iterable, List, Tuple, TypeVar
from numpy.linalg import solve
from scipy.spatial import distance_matrix
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .diffusionmap import diffusionMaps, affinity, affinity, logsumexp
from .hodgedecomp import lexsort_rows,triangle_list,gradop,divop,curlop,laplacian0,potential,grad,div,curl
from .dimensionreduction import run_pca

V = TypeVar('V')

def edges_on_path(path: List[V]) -> Iterable[Tuple[V, V]]:
    return zip(path, path[1:])


def gscale(X:np.ndarray) -> np.ndarray:
    assert(X.all()>=0)
    div_ = np.divide(X.T, np.apply_along_axis(lambda x:np.exp(np.mean(np.log(x))), 1, X)).T
    scale_ = np.apply_along_axis(np.median,0, div_)
    sc = StandardScaler(with_mean=False)
    sc.fit(X)
    sc.scale_ = scale_

    return sc.transform(X)

def knn(x,k=2):
  r = pd.Series(x).rank()
  return (r <= k+1) & (r > 1)


#alt. mat. -> networkx graph
def graph_altmat(A:np.ndarray, tol:float=1e-7) -> nx.DiGraph:
  nA = copy.deepcopy(A)
  nA[nA<tol] = 0
  G = nx.from_numpy_matrix(nA, parallel_edges=False, create_using=nx.DiGraph)
  return G


# networkx graph -> alt. mat.
def as_altmat(g:nx.DiGraph, weight_attr:str="weight") -> np.ndarray:
    if len(g.edges())==0:
     return np.zeros((len(g.nodes()), len(g.nodes())))
    ## Here I need keep the negative edges
    h = igraph.Graph.from_networkx(g)
    A = h.get_adjacency_sparse(attribute=weight_attr) ## keep the negatives
    #A = nx.adjacency_matrix(g, weight=weight_attr) ## this only gives positives
    return A - A.T


def randomalt(n:int, p:list=[0.9,0.05,0.05]) -> np.ndarray:
  np.random.seed(2022)
  A = np.random.choice([0,1,2],n*n, p =p).reshape((n,n))
  return A - A.T


def randomdata(m:int, n:int, p:list=[0.9,0.05,0.05]) -> np.ndarray:
  np.random.seed(2022)
  A = np.random.choice([0,1,2],m*n, p=p).reshape((m,n))
  return A

#def linkGraphEdges(A, W, k):
#  """
#  store k nearest neighbores of diffusion for further use
#  """
#  nei = np.apply_along_axis(knn,0,W, k)
#  A[~(nei|nei.T)] = 0
#  return graph_altmat(A).edges()
##endf

def adjedges(A, W, k=4):
    """
    From adjacency matrix & distance matrix W to knn edges

    Parameter
    ------------
    A: adjacency matrix
    W: distances
    k: nn k neighbores for edges

    Return
    -----------
    networkx.edges
    """
    nW = copy.deepcopy(W)
    nA = copy.deepcopy(A)

    nei = np.apply_along_axis(knn,0,nW, k)
    nA[~(nei|nei.T)] = 0
    kedges = graph_altmat(nA).edges()
    return kedges
#endf adjedges

#" dm cells x dimensions
def diffusionGraphDM(dm, roots,k=11,ndc=40,s=1,j=7,lmda=1e-4,sigma=None, verbose=False):
  """
  transition matrix is calculate by eigen decomposition outcome

  # M' = D^{1/2} P D^{-1/2}
  # S is the eigen vectors of M'
  psi = D^{-1/2} S
  phi = S^T D^{1/2}
    \begin{aligned}
    M^s & = Q \Lambda Q^{-1}Q \Lambda Q^{-1}\cdots Q \Lambda Q^{-1}\\
    & = Q \Lambda^s Q^{-1}\\
    & = D^{-1/2} S \Lambda^s S^T D^{1/2} \\
    & = \psi \cdot \Lambda^s \cdot \phi\\
    \end{aligned}

  Parameter
  -------------
  ndc: number of diffusion components using
  npc: number of principal components using

  Return
  --------------
  Dictionay d
  d['g']: The diffusion graph
  d['A']: Graph full adjacency matrix
  d['W']: Distances
  """

  if all(np.array(roots)==False):
    raise Exception("there should but some True cells as root")

  if verbose:
    print(datetime.now(), "distance_matrix")
  # Euclid distance matrix
  R = distance_matrix(dm, dm)
  if verbose:
    print(datetime.now(), "Diffusionmaps: ")
  d = diffusionMaps(R,j,sigma) #0:Psi, 1:Phi, 2:eig
  print("done.")
  # Diffusion distance matrix
  mm = d['psi']@np.diag(np.power(d['eig'], s))[:, 1:(ndc+1)]
  W = distance_matrix(mm, mm)
  # Transition matrix at t=s
  M = d['psi']@np.diag(np.power(d['eig'], s))@d['phi'].T
  # Set potential as density at time t=s
  #print(M)
  u = np.mean(M[roots,:], axis = 0)
  # Set potential u=-log(p) where p is density at time t=s
  #-------names(u) = colnames(X)
  # approximated -grad (related to directional deriv.?)
  A = np.subtract.outer(u,u)
  #P = with(d,Psi@diag(colSums(outer(1:10,eig,`^`)))@t(Phi))
  #P = with(d,Psi[,1]@t(Psi[1,]))
  #P = with(d,Psi @ diag(eig-1) @ diag(sapply(eig,function(x) sum(x^seq(0,100)))) @ t(Phi)) # totalflow
  #A = P-t(P)
  # divergence of fully-connected diffusion graph

  OA = copy.deepcopy(A)
  g_o = graph_altmat(A)

  #return(g_o)
  if verbose:
    print(datetime.now(), "Rewiring: ")
  div_o = div(g_o)
  # u_o = drop(potential(g_o)) # time consuming
  # k-NN graph using diffusion distance
  nei = np.apply_along_axis(knn,0,W, k)
  A[~(nei|nei.T)] = 0


  ## For test
  #csv_dir = "/home/sz753404/data/git_code/trajectory-outlier-detection-flow-embeddings/util"
  #A = np.array(pd.read_csv(f"{csv_dir}/A.csv", index_col=0))

  g = graph_altmat(A)
  # Pulling back the original potential using pruned graph
  # Lgi = MASS.ginv(as.matrix(laplacian0(g)))
  # div_s = glmnet.glmnet(Lgi,u_o,alpha=0.5,lmda=lmda).beta
  # igraph.E(g).weight = gradop(g)@Lgi@div_s
  # Pulling back the original divergeggnce using pruned graph


  if verbose:
    print(datetime.now(), "edge weight...")
  edge_weight = solve(
    divop(g).T@divop(g) + lmda * np.diag([1]*len(g.edges())),
    -gradop(g)@div_o,
  )
  attrw_dict = {(x[0], x[1]):{"weight":y} for x,y in zip(g.edges(), edge_weight)}
  nx.set_edge_attributes(g, attrw_dict)

  if verbose:
    print(datetime.now(), "grad...")
  edge_weight = grad(g, weight_attr='weight')
  attrw_dict = {(x[0], x[1]):{"weight":y} for x,y in zip(g.edges(), edge_weight)}
  nx.set_edge_attributes(g, attrw_dict)

  # drop edges with 0 weights and flip edges with negative weights
  g = graph_altmat(as_altmat(g, 'weight'))

  if verbose:
    print(datetime.now(), "ddhodge done.")
  print("done.")
  attru_dict = {x:{"u":y} for x,y in zip(g.nodes(), potential(g))}
  nx.set_node_attributes(g, attru_dict)

  attrv_dict = {x:{"div":y} for x,y in zip(g.nodes(), div(g))}
  nx.set_node_attributes(g, attrv_dict)

  attrvo_dict = {x:{"div_o":y} for x,y in zip(g.nodes(), div_o)}
  nx.set_node_attributes(g, attrvo_dict)

  return {"g":g, 'A':OA, "W":W, "psi":d["psi"], "phi":d['phi'], "eig":d['eig'], "dm":dm}
#endf diffusionGraphDM


## X: column observations,row features
def diffusionGraph(X,roots,k=11,npc=None,ndc=40,s=1,j=7,lmda=1e-4,sigma=None, verbose=False):
  """
  Parameter
  -------------
  ndc: number of diffusion components using
  npc: number of principal components using
  """
  #print("X:", X)
  #print("roots:", roots)
  #print("k:", k)
  #print("npc:", npc)
  #print("ndc:", ndc)
  #print("s:", s)
  #print("j:", j)


  print("Normalization: ")
  Y = np.log(gscale(X+0.5)).T
  print("done.")
  print("Pre-PCA: ")
  npc = min(100, Y.shape[0]-1, Y.shape[1]-1) if not npc else min(100, Y.shape[0]-1, Y.shape[1] -1, npc)
  pc = run_pca(Y, npc)

  dic = diffusionGraphDM(pc, roots=roots,k=k,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma, verbose=verbose)
  return dic
#endf diffusionGraph



