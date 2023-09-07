from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def meta_cross_dotplot(adata, cluster='celltype', sample='time', label=True, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    count_series = adata.obs.groupby([cluster, sample]).size()
    new_df = count_series.to_frame(name = 'size').reset_index()
    # convert from multi index to pivot
    constitution = new_df.pivot(index=cluster, columns=sample)['size']
    # convert to %time (but could be modified to show different things instead
    perc_clust = np.array((constitution.T / np.sum(constitution.T, axis=0)))
    # keep track of the time, cluster IDs so we can use them for plotting


    x, y = np.indices((constitution.shape))
    x = x.flatten() + 0.5
    y = y.flatten() + 0.5

    ax.scatter(x, y, s=np.array(constitution).ravel(), **kwargs)
    x_ticks = np.arange(constitution.shape[0]) + 0.5
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [constitution.index[idx] for idx, _ in enumerate(x_ticks)],
        rotation=90,
        ha='center',
        minor=False,
    )
    y_ticks = np.arange(constitution.shape[1]) + 0.5
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [constitution.columns[idx] for idx, _ in enumerate(y_ticks)], minor=False
    )
    if label:
        ax.set_xlabel(cluster)
        ax.set_ylabel(sample)
