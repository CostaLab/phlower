
# PHLOWER<img src="https://github.com/CostaLab/phlower/blob/main/figures/phlower_logo.svg" align="right" width="120" alt='logo'/>

decom**P**osition of the **H**odge **L**aplacian for inferring traject**O**ries from flo**W**s of c**E**ll diffe**R**entiation

### System Requirements
`PHLOWER` has been tested with the following OS or virtual environment:

	- macOS Sonoma 14.5
	- Linux 4.18.0/5.15.0
	- anaconda 4.12.0

### Installation

#### system dependencies



macOS:
- suite-sparse (>=7.8.2)
- graphviz (>=12.1.2)

```bash

  1. brew install suite-sparse
  2. brew install graphviz

  Manually install pygraphviz:
    export PATH=$(brew --prefix graphviz):$PATH
    export CFLAGS="-I $(brew --prefix graphviz)/include"
    export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    pip install pygraphviz
```

debian:
- libsuitesparse-dev (>=1:5.10)
- graphviz (>=2.42.2)
- libgraphviz-dev (>=2.42.2)
  
```bash
  1. apt install libsuitesparse-dev
  2. apt install graphviz libgraphviz-dev
```
conda:

- conda-forge::suitesparse(>=5.10.1)
- graphviz (>=7.1.0)
- pygraphviz (>=1.11)

```bash
  1. conda install conda-forge::suitesparse
  2. conda install conda-forge::python-graphviz
```

#### Python dependencies
We have tested python version `3.9.0`, `3.10.8`, `3.10.14`, `3.11.0`, `3.11.5`, `3.12.0`.

  - python (>=3.9.0)
  - numpy (>=1.23.5)
  - matplotlib (>=3.9.1)
  - seaborn (>=0.13.2)
  - networkx (>=2.8.8)
  - pydot (>=1.4.2)	
  - igraph (>=0.10.5)
  - scikit-learn (>=1.5.1)
  - scipy (>=1.14.0)
  - pandas (>=2.2.3)
  - plotly (>=5.23.0)
  - tqdm (>=4.65.0)
  - leidenalg (>= 0.9.1)
  - python-louvain (>=0.16)
  - colorcet(>=3.0.1)
  - umap-learn (>=0.5.5)
  - scikit-sparse (>=0.4.8)
  - scanpy (>=1.9.3)
  - adjustText (>=0.8)
  - pygraphviz (>=1.11)
  - gudhi (>=3.10.1)
  - magic-impute (>=3.0.0)
  - anndata (>=0.9.2)

#### install from pypi (0.1.3)
```bash
pip install phlowerpy

```

#### install newest phlower version
```bash
git clone https://github.com/CostaLab/phlower.git
cd phlower
pip install .
```

```python
import phlower
```
### Tutorial

#### Demo
A small scRNA-seq data [Fibroblast to Neuron](https://phlower.readthedocs.io/en/latest/notebooks/fib2neuron.html).

#### Kidney in the paper
10X multiome data [Kidney](https://phlower.readthedocs.io/en/latest/notebooks/kidney.html).

Regulators detection example please check [Regulators](https://phlower.readthedocs.io/en/latest/regulators.html).


### Data info
The processed data have been deposited at: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13860460.svg)](https://doi.org/10.5281/zenodo.13860460).



### reproducibility
https://github.com/CostaLab/phlower-reproducibility

### Reference

A preprint is available at [https://doi.org/10.1101/2024.10.01.613179](https://doi.org/10.1101/2024.10.01.613179)
