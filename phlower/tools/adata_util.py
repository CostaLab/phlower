import pandas as pd
import numpy as np
from anndata import AnnData
def magic_adata(adata:AnnData, random_state=2022, iscopy=False, verbose=True, **kwargs):
    import magic

    adata = adata.copy() if iscopy else adata
    magic_operator = magic.MAGIC(random_state=random_state, **kwargs)

    X =pd.DataFrame(adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X, index=adata.obs_names, columns=adata.var_names)
    X_ = magic_operator.fit_transform(X, genes=adata.var_names)

    adata.X = np.array(X_)

    return adata if iscopy else None
#endf magic_adata


def subset_adata_obs(adata:AnnData, obs_key:str, obs_subset:list, iscopy=False, verbose=True):
    """
    Subset AnnData object by obs key and obs subset
    """
    adata = adata.copy() if iscopy else adata
    idx = np.where(adata.obs[obs_key].isin(obs_subset))[0]
    adata = adata[idx, :]

    return adata if iscopy else None

