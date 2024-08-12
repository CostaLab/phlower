import sklearn
import copy
import igraph
import scipy
import scipy.spatial
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain
from typing import Iterable, List, Tuple, TypeVar, Union
from scipy.sparse.linalg import spsolve
from scipy.spatial import distance_matrix
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .diffusionmap import diffusionMaps, affinity, logsumexp
from .hodgedecomp import lexsort_rows,triangle_list,gradop,divop,curlop,laplacian0,potential,grad,div,curl, div_adj
from .dimensionreduction import run_pca


from ..util import test_cholesky, has_islands

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
def graph_altmat(A:Union[csc_matrix, csr_matrix, np.ndarray], tol:float=1e-7) -> nx.DiGraph:
  nA = copy.deepcopy(A)
  #nA = nA.todense()

  if isinstance(nA, np.ndarray):
    nA[nA<tol] = 0
    return nx.from_numpy_matrix(nA, parallel_edges=False, create_using=nx.DiGraph)

  nA.data[nA.data < tol] = 0
  nA.eliminate_zeros()
  #--------nA[nA<tol] = 0 ----------------------
  G = nx.from_scipy_sparse_array(nA, parallel_edges=False, create_using=nx.DiGraph)
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

def adjedges(A:Union[csc_matrix, csr_matrix, np.ndarray], W, k=4):
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
    zerofilter = ~(nei|nei.T)
    if isinstance(nA, np.ndarray):
        pass
    elif isinstance(nA, csc_matrix):
        zerofilter = csc_matrix(zerofilter)
    elif isinstance(nA, csr_matrix):
        zerofilter = csr_matrix(zerofilter)

    nA[zerofilter] = 0

    if scipy.sparse.issparse(nA):
        nA.eliminate_zeros()
    kedges = graph_altmat(nA).edges()
    return kedges
#endf adjedges

#" dm cells x dimensions
def diffusionGraphDM(dm, roots,k=11,ndc=40,s=1,j=7,lmda=1e-4,sigma=None, verbose=False, lstsq_method="lstsq"):
  """
  transition matrix is calculate by eigen decomposition outcome

  # M' = D^{1/2} P D^{-1/2}
  # S is the eigen vectors of M'
  psi = D^{-1/2} S
  phi = S^T D^{1/2}
    \\begin{aligned}
    M^s & = Q \Lambda Q^{-1}Q \Lambda Q^{-1}\cdots Q \Lambda Q^{-1}\\
    & = Q \Lambda^s Q^{-1}\\
    & = D^{-1/2} S \Lambda^s S^T D^{1/2} \\
    & = \psi \cdot \Lambda^s \cdot \phi\\
    \end{aligned}

  Parameters
  -------------
  dm: numpy array
    dimension reduction of diffusion map input
  roots: list
    list of bool values if they are root nodes
  k: int
    number of nearest neighbors for diffusion map
  npc: int
    number of principal components using
  ndc: int
    number of diffusion components using
  s: int
    number of stepes for diffusion map  random walk(default 1)
  lmda: float
    regularization parameter for ddhodge(default 1e-4)
  sigma: float
    sigma for gaussian kernel(default None)
  verbose: bool
    print out progress(default False)
  lstsq_method: str
    method for least square solver,  "lstsq" or "lsqr", "lsmr", cholesky (default "lstsq")

  Return
  ---------------
  dic: dict
     d['g']: The diffusion graph\n
     d['A']: Graph full adjacency matrix\n
     d['W']: Diffusion Distances\n
     d['psi']: Diffusion Map right eigenvectors\n
     d['phi']: Diffusion Map left eigenvectors\n
     d['eig']: Diffusion Map eigenvalues\n
     d['dm']: dimension reduction input of diffusion map
  """

  if all(np.array(roots)==False):
    raise Exception("there should but some True cells as root")

  if verbose:
    print(datetime.now(), "distance_matrix")
  # Euclid distance matrix
  R = distance_matrix(dm, dm)
  if verbose:
    print(datetime.now(), "Diffusionmaps: ")
  d = diffusionMaps(R,j,sigma, eig_k=ndc+1) #0:Psi, 1:Phi, 2:eig
  print("done.")
  # Diffusion distance matrix
  mm = d['psi']@np.diag(np.power(d['eig'], s))[:, 1:(ndc+1)]

  if verbose:
    print(datetime.now(),"diffusion distance:")
  W = distance_matrix(np.array(mm), np.array(mm))
  if verbose:
    print(datetime.now(),"transition matrix:")
  # Transition matrix at t=s
  M = d['psi']@np.diag(np.power(d['eig'], s))@d['phi'].T
  # Set potential as density at time t=s
  #print(M)
  u = np.mean(M[roots,:], axis = 0)
  del M
  # Set potential u=-log(p) where p is density at time t=s
  #-------names(u) = colnames(X)
  # approximated -grad (related to directional deriv.?)
  A = scipy.sparse.csr_matrix(np.subtract.outer(u,u))
  #P = with(d,Psi@diag(colSums(outer(1:10,eig,`^`)))@t(Phi))
  #P = with(d,Psi[,1]@t(Psi[1,]))
  #P = with(d,Psi @ diag(eig-1) @ diag(sapply(eig,function(x) sum(x^seq(0,100)))) @ t(Phi)) # totalflow
  #A = P-t(P)
  # divergence of fully-connected diffusion graph

  if verbose:
    print(datetime.now(),"graph from A")

  OA = copy.deepcopy(A)
  #g_o = graph_altmat(A)

  #return(g_o)
  if verbose:
    print(datetime.now(), "Rewiring: ")

  if verbose:
    print(datetime.now(), "div(g_o)...")
  #div_o = div(g_o)
  div_o = div_adj(A)
  #del g_o
  # u_o = drop(potential(g_o)) # time consuming
  # k-NN graph using diffusion distance


  nei = np.apply_along_axis(knn,0,W, k)
  zerofilter = scipy.sparse.csr_matrix(~(nei|nei.T))
  A[zerofilter] = 0
  g = graph_altmat(A)

  while False: ## if A is not connected
    if nx.is_connected(g.to_undirected()):
        if verbose:
            print(datetime.now(), "connected graph k=",k)
        break
    else:
        if verbose:
            print(datetime.now(), "not connected graph k=",k)
    A = copy.deepcopy(OA)
    k = k+1
    nei = np.apply_along_axis(knn,0,W, k)
    zerofilter = scipy.sparse.csr_matrix(~(nei|nei.T))
    A[zerofilter] = 0
    A.eliminate_zeros()
    if verbose:
      print(datetime.now(), "create graph...")
    g = graph_altmat(A)
    if k>100:
        raise Exception("k is too large")



  # Pulling back the original potential using pruned graph
  # Lgi = MASS.ginv(as.matrix(laplacian0(g)))
  # div_s = glmnet.glmnet(Lgi,u_o,alpha=0.5,lmda=lmda).beta
  # igraph.E(g).weight = gradop(g)@Lgi@div_s
  # Pulling back the original divergeggnce using pruned graph

  if verbose:
    print(datetime.now(), "edge weight...")
  ## sparse csc matrix
  a = divop(g).T@divop(g) + lmda * scipy.sparse.diags([1]*len(g.edges()), 0, format="csc")
  b = -gradop(g)@div_o

  #cg_ret = -1
  if verbose:
    print(datetime.now(), "cholesky solve ax=b...")
  ret = test_cholesky(a, verbose=verbose) #cholesky to solve ax=b
  if ret:
    edge_weight = ret(b)
    #edge_weight, cg_ret = scipy.sparse.linalg.cg(a, b, tol=1e-6)
  else:
    if verbose:
        print(datetime.now(), "cholesky failed, use LS instead...")
    edge_weight = scipy.sparse.linalg.spsolve(
      a,
      b,
    )
  del a,b



#scipy.sparse.linalg.lsqr
#scipy.sparse.linalg.lsmr: probably faster

  attrw_dict = {(x[0], x[1]):{"weight":y} for x,y in zip(g.edges(), edge_weight)}
  nx.set_edge_attributes(g, attrw_dict)

  if verbose:
    print(datetime.now(), "grad...")
  edge_weight = grad(g, weight_attr='weight', lstsq_method=lstsq_method)
  attrw_dict = {(x[0], x[1]):{"weight":y} for x,y in zip(g.edges(), edge_weight)}
  nx.set_edge_attributes(g, attrw_dict)

  # drop edges with 0 weights and flip edges with negative weights
  g = graph_altmat(as_altmat(g, 'weight'))

  if verbose:
    print(datetime.now(), "potential.")
  attru_dict = {x:{"u":y} for x,y in zip(g.nodes(), potential(g, method=lstsq_method))}
  nx.set_node_attributes(g, attru_dict)
  if verbose:
    print(datetime.now(), "ddhodge done.")
  print("done.")

  attrv_dict = {x:{"div":y} for x,y in zip(g.nodes(), div(g))}
  nx.set_node_attributes(g, attrv_dict)

  attrvo_dict = {x:{"div_o":y} for x,y in zip(g.nodes(), div_o)}
  nx.set_node_attributes(g, attrvo_dict)

  return {"g":g, 'A':OA, "W":W, "psi":d["psi"], "phi":d['phi'], "eig":d['eig'], "dm":dm}
#endf diffusionGraphDM


## X: column observations,row features
def diffusionGraph(X,roots,k=11,npc=None,ndc=40,s=1,j=7,lmda=1e-4,sigma=None, verbose=False, lstsq_method="lstsq"):
  """
  Parameters
  -------------
  X: numpy array
    column observations,row features
  roots: list
    list of bool values if they are root nodes
  k: int
    number of nearest neighbors for diffusion map
  npc: int
    number of principal components using
  ndc: int
    number of diffusion components using
  s: int
    number of stepes for diffusion map  random walk(default 1)
  lmda: float
    regularization parameter for ddhodge(default 1e-4)
  sigma: float
    sigma for gaussian kernel(default None)
  verbose: bool
    print out progress(default False)
  lstsq_method: str
    method for least square solver,  "lstsq" or "lsqr", "lsmr" (default "lstsq")

  Return
  ---------------
  dic: dict
     d['g']: The diffusion graph
     d['A']: Graph full adjacency matrix
     d['W']: Diffusion Distances
     d['psi']: Diffusion Map right eigenvectors
     d['phi']: Diffusion Map left eigenvectors
     d['eig']: Diffusion Map eigenvalues
     d['dm']: dimension reduction input of diffusion map

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

  dic = diffusionGraphDM(pc, roots=roots,k=k,ndc=ndc,s=s,j=j,lmda=lmda,sigma=sigma, verbose=verbose, lstsq_method="lstsq")
  return dic
#endf diffusionGraph



