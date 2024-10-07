
# PHLOWER<img src="https://github.com/CostaLab/phlower/blob/main/figures/phlower_logo.svg" align="right" width="120" alt='logo'/>

decom**P**osition of the **H**odge **L**aplacian for inferring traject**O**ries from flo**W**s of c**E**ll diffe**R**entiation

### Installation

#### system dependences
```bash
macos:
  1. brew install suite-sparse
  2. brew install graphpviz

  Manually install pygraphviz:
    export PATH=$(brew --prefix graphviz):$PATH
    export CFLAGS="-I $(brew --prefix graphviz)/include"
    export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    pip install pygraphviz

debian:
  1. apt install libsuitesparse-dev
  2. apt install graphviz libgraphviz-dev

conda:
  1. conda install conda-forge::suitesparse
  2. conda install conda-forge::python-graphviz
```

#### install from pypi
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
Tutorials exampled by a small scRNA-seq data [Fibroblast to Neuron](https://phlower.readthedocs.io/en/latest/notebooks/fib2neuron.html) and a 10X multiome data [Kidney](https://phlower.readthedocs.io/en/latest/notebooks/kidney.html).


### Data info
https://doi.org/10.5281/zenodo.13860460


### Reference

A preprint is available at: [https://doi.org/10.1101/2024.10.01.613179](https://doi.org/10.1101/2024.10.01.613179)
