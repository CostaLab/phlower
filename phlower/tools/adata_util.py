def magic_adata(adata, n_pcs=50, n_neighbors=15, n_jobs=1, random_state=0, iscopy=False, verbose=True):
    import magic

    adata = adata.copy() if iscopy else adata
    magic_operator = magic.MAGIC()

    X =pd.DataFrame(adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X, index=adata.obs_names, columns=adata.var_names)
    X_ = magic_operator.fit_transform(X, genes=adata.var_names)

    adata.X = np.array(X_)

    return adata if iscopy else None
#endf magic_adata
