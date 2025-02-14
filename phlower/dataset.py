import os
from urllib.request import urlretrieve
from anndata import read_h5ad
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def _data_download(url, file_path=None, folder="data", verbose=True):
    file_path = ntpath.basename(url) if file_path is None else file_path
    file_path = os.path.join(folder, file_path)
    if verbose:
        print("Downloading data to " + file_path)

    if not os.path.exists(file_path):
        if not os.path.exists("data/"):
            os.mkdir("data")

        ## downloading
        urlretrieve(url, file_path)
    return file_path

def fib2neuro(
        url = 'https://costalab.ukaachen.de/open_data/PHLOWER/fib2neuron.h5ad',
        fname = 'fib2neuro.h5ad',
        folder = 'data',
        verbose = True,
        ):
    fpath = _data_download(url, fname, folder=folder, verbose=verbose)
    adata = read_h5ad(fpath)
    adata.var_names_make_unique()
    return adata

def kidney(
        url = 'https://costalab.ukaachen.de/open_data/PHLOWER/kidney.h5ad',
        fname = 'kidney.h5ad',
        folder = 'data',
        verbose = True,
        ):
    fpath = _data_download(url, fname, folder=folder, verbose=verbose)
    adata = read_h5ad(fpath)
    adata.var_names_make_unique()
    return adata


def neurogenesis(
        url = 'https://costalab.ukaachen.de/open_data/PHLOWER/neurogenesis.h5ad',
        fname = 'neurogenesis.h5ad',
        folder = 'data',
        verbose = True,
        ):
    fpath = _data_download(url, fname, folder=folder, verbose=verbose)
    adata = read_h5ad(fpath)
    adata.var_names_make_unique()
    return adata


def pancreas(
        url = 'https://costalab.ukaachen.de/open_data/PHLOWER/pancreas.h5ad',
        fname = 'pancreas.h5ad',
        folder = 'data',
        verbose = True,
        ):
    fpath = _data_download(url, fname, folder=folder, verbose=verbose)
    adata = read_h5ad(fpath)
    adata.var_names_make_unique()
    return adata

def dla10(
        url = 'https://costalab.ukaachen.de/open_data/PHLOWER/DLA_10_TreeData.mat',
        fname = 'DLA_10_TreeData.mat',
        folder = 'data',
        verbose = True,
        ):
    fpath = _data_download(url, fname, folder=folder, verbose=verbose)
    from scipy.io import loadmat
    mat = loadmat(fpath)
    return mat
#endf dla10







def human_hematopoisis(
        url = '',
        fname = 'human_hematopoisis.h5ad',
        folder = 'data',
        verbose = True,
        ):
    fpath = _data_download(url, fname, folder=folder, verbose=verbose)
    adata = read_h5ad(fpath)
    adata.var_names_make_unique()
    return adata


