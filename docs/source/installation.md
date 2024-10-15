
## Installation

##### system dependences
```bash
macos:
  1. brew install suite-sparse
  2. brew install graphviz

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

##### install from pypi
```bash
pip install phlowerpy
```


##### install newest phlower version
```bash
git clone https://github.com/CostaLab/phlower.git
cd phlower
pip install .
```


##### Run

  ```python
  import phlower
  ```
