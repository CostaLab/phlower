import scipy
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import linalg


def logsumexp(x):
    maxx=max(x)
    return maxx + np.log(np.sum(np.exp(x - maxx)))

def affinity(R, k=7, sigma=None, log=False, normalize=False):
    """
    Gaussian affinity matrix constructor
    W = exp(-r_{ij}^2/sigma)

    Parameters
    -----------
    R: np.array
        symmetric matrix(positive semi-definite); Distance matrix
    k: int
        number of neighbors in adaptive-scaling, ignore if sigma is not None
    log: bool
        transform the affinity by logrithm or not
    normalize: bool
        return transition matrix
    """
    def top_k(lst, k=1):
        assert(len(lst) >k)
        return np.partition(lst, k)[k]
    R = np.array(R)
    if not sigma:
        s = [top_k(R[:, i], k=k)  for i in range(R.shape[1])]
        S = np.sqrt(np.outer(s, s))
    else:
        S = sigma
    logW = -np.power(np.divide(R, S), 2)

    if normalize:
        #sweep(logW,1,apply(logW,1,logsumexp)) #wrong, need to divide, FUN='/'
        denominator = [logsumexp(logW[i,:]) for i in range(logW.shape[0])]
        logW = np.divide(logW.T, denominator).T
    if log:
        return logW
    return np.exp(logW)


def  diffusionMaps(R,k=7,sigma=None, verbose=False, eig_k=100):
    """
    Diffusion map(Coifman, 2005)
    https://en.wikipedia.org/wiki/Diffusion_map

    Parameters
    ----------
    R: np.array
        symmetric matrix(positive semi-definite); Distance matrix
    k: int
        number of neighbors in adaptive-scaling
    sigma: float
        for isotropic diffussion
    Return
    ----------
    dic:
        psi: scipy.sparse.csr_matrix
            right eigvector of P = D^{-1/2} * evec
        phi: scipy.sparse.csr_matrix
            left eigvector of P = D^{1/2} * evec
        eig: np.array
            eigenvalues
    """
    k=k-1 ## k is R version minus 1 for the index
    if verbose:
        print(datetime.now(), "Affinity matrix construction...")

    logW = affinity(R,k,sigma,log=True,normalize=False)
    rs = np.exp([logsumexp(logW[i,:]) for i in range(logW.shape[0])]) ## dii=\sum_j w_{i,j}
    #L=D-W, dii = \sum_j w_{ij}
    D = scipy.sparse.diags(np.sqrt(rs), 0, format='csr')        ## D^{1/2}
    Dinv = scipy.sparse.diags(1/np.sqrt(rs),0, format='csr')   ##D^{-1/2}
    #dela porte etal. An Introduction to Diffusion Maps
    # normalized W: P = D^{-1} W
    # P' = D^{1/2}PD^{-1/2} =  D^{-1/2} W D^{-1/2}
    Ms = scipy.sparse.csr_matrix(Dinv @ np.exp(logW) @ Dinv)##
    ## https://jlmelville.github.io/smallvis/spectral.html row normalisation
    #e = eigen(Ms,symmetric=TRUE)

    if verbose:
        print(datetime.now(), "Eigen decomposition...")

    e = linalg.eigsh(Ms, k=eig_k) ## eigen decomposition of P'
    #e = np.linalg.eigh(Ms) ## eigen decomposition of P'
    evalue= e[0][::-1]
    evec =np.flip(e[1], axis=1)
    s =(np.sum(np.sqrt(rs) * evec[:,0])) # scaling
    # Phi is orthonormal under the weighted inner product
    #0:Psi, 1:Phi, 2:eig
    dic = {'psi':scipy.sparse.csr_matrix(s * Dinv@evec), 'phi': scipy.sparse.csr_matrix((1/s)*D@evec), "eig": evalue}
    return dic
    #return s * Dinv@evec, (1/s)*D@evec, evalue
