
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

#### install phlower
```bash
git clone git@github.com:CostaLab/phlower.git
pip install phlower
```

```python
import phlower
```

