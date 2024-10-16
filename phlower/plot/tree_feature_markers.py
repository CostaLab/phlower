import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union, List
from anndata import AnnData
from ..tools.tree_feature_markers import branch_heatmap_matrix
from ..tools.tree_utils import TF_to_genes

def plot_rank_gene_group(adata,
                         name='markers_1_21_vs_0_17.2_21',
                         n_genes=10,
                         **kwargs):
    """
    Parameters
    ----------
    adata: AnnData
        An AnnData object.
    name: str
        The name of the differential expression analysis result.
    n_genes: int
        The number of genes to plot.
    """
    if name not in adata.uns:
        raise ValueError('Name not found in adata.uns, please run `sc.tl.tree_nodes_markers` first.')
    sc.pl.rank_genes_groups(adata.uns[name], n_genes=n_genes, **kwargs)


#def plot_rank_gene_group_dotplot(adata, name='markers_1_21_vs_0_17.2_21', n_genes=10, **kwargs):
#    if name not in adata.uns:
#        raise ValueError('Name not found in adata.uns, please run `sc.tl.tree_nodes_markers` first.')
#    sc.pl.rank_genes_groups_dotplot(adata.uns[name], n_genes=n_genes, **kwargs)
#


#def plot_rank_gene_group_heatmap(adata, name='markers_1_21_vs_0_17.2_21', n_genes=10, **kwargs):
#    if name not in adata.uns:
#        raise ValueError('Name not found in adata.uns, please run `sc.tl.tree_nodes_markers` first.')
#    sc.pl.rank_genes_groups_dotplot(adata.uns[name], n_genes=n_genes, **kwargs)
#


#def plot_rank_gene_group_stacked_violin(adata, name='markers_1_21_vs_0_17.2_21', n_genes=10, **kwargs):
#    if name not in adata.uns:
#        raise ValueError('Name not found in adata.uns, please run `sc.tl.tree_nodes_markers` first.')
#    sc.pl.rank_genes_groups_dotplot(adata.uns[name], n_genes=n_genes, **kwargs)
#

def volcano(df: pd.DataFrame,
            gene_column='names',
            log2fc_column='logfoldchanges',
            pval_column='pvals',
            genes_to_highlight=None,
            log2fc_threshold: Union[List[int], int]=2,
            pval_threshold=0.01,
            show_legend=True,
            up_color='red',
            down_color='blue',
            not_sig_color='grey',
            is_adjust_text=False,
            ax=None,
            swap_axes=False,
            text_up_down="all", ## all, up, down
            text_size=8,
            **kwargs
            ):
    """
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with columns of gene names, log2 fold changes, and p-values.
    gene_column: str
        The column name of gene names.
    log2fc_column: str
        The column name of log2 fold changes.
    pval_column: str
        The column name of p-values.
    genes_to_highlight: list
        A list of genes to highlight.
    log2fc_threshold: int or list
        The threshold of log2 fold changes to highlight.
    pval_threshold: float
        The threshold of p-values to highlight.
    show_legend: bool
        Whether to show the legend.
    up_color: str
        The color of up-regulated genes.
    down_color: str
        The color of down-regulated genes.
    not_sig_color: str
        The color of non-significant genes.
    is_adjust_text: bool
        whether to adjust text to avoid overlap, very slow
    ax: matplotlib.axes.Axes
        A matplotlib axes object.
    text_size: int
        The size of text.
    """
    from adjustText import adjust_text
    nplog10_ = lambda x: -np.log10(x)

    if ax is None:
        ax = plt.gca()

    log2fc_threshold_neg = -1 * log2fc_threshold
    log2fc_threshold_posi = log2fc_threshold_neg
    if not isinstance(log2fc_threshold, int):
        assert len(log2fc_threshold) == 2
        log2fc_threshold_neg = log2fc_threshold[0]
        log2fc_threshold_posi = log2fc_threshold[1]
        assert log2fc_threshold_neg < log2fc_threshold_posi


    df.index = df[gene_column]
    y = df[pval_column].apply(lambda x:nplog10_(x))
    # avoid infinite value
    y[y == np.inf] = y[y != np.inf].max()
    if swap_axes:
        ax.scatter(y=df[log2fc_column],x=y,s=1,label="Not significant", color=not_sig_color, **kwargs)
    else:
        ax.scatter(x=df[log2fc_column],y=y,s=1,label="Not significant", color=not_sig_color, **kwargs)

    # highlight down- or up- regulated genes
    down = df[(df[log2fc_column]<=log2fc_threshold_neg)&(df[pval_column]<=pval_threshold)]
    up = df[(df[log2fc_column]>=log2fc_threshold_posi)&(df[pval_column]<=pval_threshold)]
    y =down[pval_column].apply(lambda x:nplog10_(x))
    y[y == np.inf] = y[y != np.inf].max()
    if swap_axes:
        ax.scatter(y=down[log2fc_column],x=y,s=3,label="Down-regulated",color=down_color, **kwargs)
    else:
        ax.scatter(x=down[log2fc_column],y=y,s=3,label="Down-regulated",color=down_color, **kwargs)
    y = up[pval_column].apply(lambda x:nplog10_(x))
    y[y == np.inf] = y[y != np.inf].max()
    if swap_axes:
        ax.scatter(y=up[log2fc_column],x=y,s=3,label="Up-regulated",color=up_color, **kwargs)
    else:
        ax.scatter(x=up[log2fc_column],y=y,s=3,label="Up-regulated",color=up_color, **kwargs)

    texts = []
    if genes_to_highlight is not None:
        genes_to_highlight = set(genes_to_highlight)
        texts = [plt.text(df.loc[gene,log2fc_column], nplog10_(df.loc[gene,pval_column]), '%s' %gene) for gene in genes_to_highlight if gene in df.index]
        for gene in genes_to_highlight:
            if gene in df.index:
                if swap_axes:
                    ax.scatter(y=df.loc[gene,log2fc_column],
                               x=nplog10_(df.loc[gene,pval_column]),
                               s=10,
                               label=gene,
                               size=text_size,
                               color='black', **kwargs)

                else:
                    ax.scatter(x=df.loc[gene,log2fc_column],
                               y=nplog10_(df.loc[gene,pval_column]),
                               s=10,
                               label=gene,
                               size=text_size,
                               color='black', **kwargs)
            else:
                print(f'Gene {gene} not found in the dataframe.')
    else:
        if text_up_down in ('all', "up"):
            for i,r in up.iterrows():
                if swap_axes:
                    texts.append(ax.text(y=r[log2fc_column],x=nplog10_(r[pval_column]),s=i, size=text_size))
                else:
                    texts.append(ax.text(x=r[log2fc_column],y=nplog10_(r[pval_column]),s=i, size=text_size))
        elif text_up_down in ("all", "down"):
            for i,r in down.iterrows():
                if swap_axes:
                   texts.append(ax.text(y=r[log2fc_column],x=nplog10_(r[pval_column]),s=i, size=text_size))
                else:
                   texts.append(ax.text(x=r[log2fc_column],y=nplog10_(r[pval_column]),s=i, size=text_size))

    if is_adjust_text and len(texts)>0:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))


    if swap_axes:
        ax.set_ylabel(f"{log2fc_column}")
        ax.set_xlabel(f"-log10 {pval_column}")
        ax.axhline(log2fc_threshold_neg,color="grey",linestyle="--")
        ax.axhline(log2fc_threshold_posi,color="grey",linestyle="--")
        ax.axvline(nplog10_(pval_threshold),color="grey",linestyle="--")
    else:
        ax.set_xlabel(f"{log2fc_column}")
        ax.set_ylabel(f"-log10 {pval_column}")
        ax.axvline(log2fc_threshold_neg,color="grey",linestyle="--")
        ax.axvline(log2fc_threshold_posi,color="grey",linestyle="--")
        ax.axhline(nplog10_(pval_threshold),color="grey",linestyle="--")

    if show_legend:
        ax.legend()


def marker_line_plot(adata: AnnData,
                     features,
                     obs_time_key,
                     is_magic_impute=False,
                     verbose=True,
                     is_scatter = False,
                     color = ['red'],
                     is_legend = True,
                     is_allinone = False,
                     ax = None, ## only allinone
                     **kwargs):
    """
    Plot the expression of a marker gene along time.

    Parameters
    ----------
    adata: AnnData
        AnnData object
    features: list
        List of marker genes
    obs_time_key: str
        Key of the time information in adata.obs
    is_magic_impute: bool
        Whether to impute the data using MAGIC
    verbose: bool
        Whether to print the progress
    is_scatter: bool
        Whether to use scatter plot
    color: list
        List of colors for each marker gene
    is_legend: bool
        Whether to show legend
    is_allinone: bool
        Whether to plot all genes in one figure
    ax: matplotlib.axes.Axes
        Axes object for only all in one
    kwargs:
        Additional arguments for line plotting
    """
    import scipy
    from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
    def smooth(x, y, xgrid):
        samples = np.random.choice(len(x), 50, replace=True)
        #print(samples)
        y_s = y[samples]
        x_s = x[samples]
        y_sm = sm_lowess(y_s,x_s, frac=1./5., it=5, return_sorted = False)
        # regularly sample it onto the grid
        y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
        return y_grid

    if color is None:
        color = ['red']


    features = [features] if isinstance(features, str) else features
    #for feature in features:
    #    if feature not in adata.var_names:
    #        print("Feature %s not found in the adata.var_names"%feature)

    features = [feature for feature in features if feature in adata.var_names]

    if len(features) == 0:
        raise ValueError("No feature found in the adata.var_names")

    #X_ = None
    if is_magic_impute:
        import magic
        magic_operator = magic.MAGIC()

        idxs = [np.where(adata.var_names == feature)[0][0] for feature in features]
        X = adata.X[:, idxs]
        X =pd.DataFrame(X.toarray() if not isinstance(X, np.ndarray) else X, index=adata.obs_names, columns=features)
        X_ = magic_operator.fit_transform(X, genes=features)
    else:
        idxs = [np.where(adata.var_names == feature)[0][0] for feature in features]
        X_ =pd.DataFrame(adata.X[:,idxs].toarray() if not isinstance(adata.X, np.ndarray) else adata.X[:, idxs], index=adata.obs_names, columns=features)

    if not obs_time_key in adata.obs.columns:
        raise ValueError(f"obs_time_key {obs_time_key} is not in adata.obs.columns")

    if is_allinone:
        if len(color) < len(features):
            import colorcet as cc
            import seaborn as sns
            color = sns.color_palette(cc.glasbey, n_colors=len(features)).as_hex()
        if ax is None:
            _, ax = plt.subplots(1,1)
        x = np.array(adata.obs[obs_time_key].values)
        for idx, feature in enumerate(features):
            y = np.array(X_.loc[:, feature])
            x_ = sorted(set(x))
            y_ = [np.mean(y[x==k]) for k in x_]
            ax.plot(x_, y_, marker='.', linestyle='-', markersize=10, color=color[idx],  label=feature, zorder=5, **kwargs)
        if len(set(x)) < 10:
            ax.set_xticks(sorted(set(x)))
        if is_legend:
            ax.legend()

    else:
        x = np.array(adata.obs[obs_time_key].values)
        for idx, feature in enumerate(features):
            _, ax = plt.subplots(1,1 )
            y = np.array(X_.loc[:, feature])
            x_ = sorted(set(x))
            y_ = [np.mean(y[x==k]) for k in x_]
            ax.plot(x_, y_, marker='.', linestyle='-', markersize=10, color=color[0],  label=feature, zorder=5, **kwargs)
            ax.set_title(feature)
            if is_scatter:
                ax.plot(x, y, 'k.', zorder=1)
            if len(set(x)) < 10:
                ax.set_xticks(sorted(set(x)))
            if is_legend:
                ax.legend()
#endf marker_line_plot





def marker_spline_plot(adata: AnnData,
                     features,
                     obs_time_key,
                     is_magic_impute=False,
                     smooth_method='lowess',
                     smooth_K=100,
                     verbose=True,
                     is_fill_conf=False,
                     is_scatter = False,
                     color = ['red'],
                     is_legend = True,
                     is_allinone = False,
                     ax = None, ## only allinone
                     seed=2022,
                     **kwargs):
    """
    Plot the expression of a marker gene along time, or pseudo time.

    Parameters
    ----------
    adata: AnnData
        AnnData object
    features: list
        List of marker genes
    obs_time_key: str
        Key of the time information in adata.obs
    is_magic_impute: bool
        Whether to impute the data using MAGIC
    smooth_method: str
        Method for smoothing the data, default: lowess
    smooth_K: int
        Number of points for smoothing
    verbose: bool
        Whether to print the progress
    is_fill_conf: bool
        Whether to fill the confidence interval
    is_scatter: bool
        Whether to use scatter plot
    color: list
        List of colors for each marker gene
    is_legend: bool
        Whether to show legend
    is_allinone: bool
        Whether to plot all genes in one figure
    ax: matplotlib.axes.Axes
        Axes object for only all in one
    kwargs:
        Additional arguments for line plotting
    """
    import scipy
    from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
    np.random.seed(seed)
    def smooth(x, y, xgrid):
        samples = np.random.choice(len(x), 50, replace=True)
        #print(samples)
        y_s = y[samples]
        x_s = x[samples]
        y_sm = sm_lowess(y_s,x_s, frac=1./5., it=5, return_sorted = False)
        # regularly sample it onto the grid
        y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
        return y_grid

    if color is None:
        color = ['red']


    features = [features] if isinstance(features, str) else features
    for feature in features:
        if feature not in adata.var_names:
            print("Feature %s not found in the adata.var_names"%feature)

    features = [feature for feature in features if feature in adata.var_names]

    if len(features) == 0:
        raise ValueError("No feature found in the adata.var_names")

    #X_ = None
    if is_magic_impute:
        import magic
        magic_operator = magic.MAGIC()
        idxs = [np.where(adata.var_names == feature)[0][0] for feature in features]
        X = adata.X[:, idxs]
        X =pd.DataFrame(X.toarray() if not isinstance(X, np.ndarray) else X, index=adata.obs_names, columns=features)
        X_ = magic_operator.fit_transform(X, genes=features)
    else:
        idxs = [np.where(adata.var_names == feature)[0][0] for feature in features]
        X_ =pd.DataFrame(adata.X[:,idxs].toarray() if not isinstance(adata.X, np.ndarray) else adata.X[:, idxs], index=adata.obs_names, columns=features)

    if not obs_time_key in adata.obs.columns:
        raise ValueError(f"obs_time_key {obs_time_key} is not in adata.obs.columns")

    if is_allinone:
        if len(color) < len(features):
            import colorcet as cc
            import seaborn as sns
            color = sns.color_palette(cc.glasbey, n_colors=len(features)).as_hex()
        if ax is None:
            _, ax = plt.subplots(1,1 )
        x = np.array(adata.obs[obs_time_key].values)
        for idx, feature in enumerate(features):
            y = np.array(X_.loc[:, feature])
            xgrid = np.linspace(x.min(),x.max())
            smooths = np.stack([smooth(x, y, xgrid) for k in range(smooth_K)]).T
            mean = np.nanmean(smooths, axis=1)
            stderr = scipy.stats.sem(smooths, axis=1)
            stderr = np.nanstd(smooths, axis=1, ddof=0)
            if is_fill_conf:
                ax.fill_between(xgrid, mean-1.96*stderr, mean+1.96*stderr, alpha=0.25, zorder=1)
            ax.plot(xgrid, mean, color=color[idx], label=feature, zorder=5,**kwargs)
            #ax.set_title(feature)
            #if is_scatter:
            #    ax.plot(x, y, 'k.')
        if len(set(x)) < 10:
            ax.set_xticks(sorted(set(x)))
        if is_legend:
            ax.legend()

    else:

        x = np.array(adata.obs[obs_time_key].values)
        for idx, feature in enumerate(features):
            _, ax = plt.subplots(1,1 )
            y = np.array(X_.loc[:, feature])
            xgrid = np.linspace(x.min(),x.max())
            smooths = np.stack([smooth(x, y, xgrid) for k in range(smooth_K)]).T
            mean = np.nanmean(smooths, axis=1)
            stderr = scipy.stats.sem(smooths, axis=1)
            stderr = np.nanstd(smooths, axis=1, ddof=0)

            if is_fill_conf:
                ax.fill_between(xgrid, mean-1.96*stderr, mean+1.96*stderr, alpha=0.25, zorder=0)
            ax.plot(xgrid, mean, color=color[0], label=feature, zorder=5, **kwargs)
            #plt.plot(xgrid, smooths, color='tomato', alpha=0.25)
            ax.set_title(feature)
            if is_scatter:
                ax.plot(x, y, 'k.', zorder=1)
            if len(set(x)) < 10:
                ax.set_xticks(sorted(set(x)))
            if is_legend:
                ax.legend()
#endf marker_sline_plot


def marker_line_pesudo(mtx,
                        features,
                        smooth_method = 'lowess',
                        is_legend = True,
                        is_allinone=False,
                        color=["red"]):

    from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
    x = [int(i.split('_')[1]) for i in mtx.columns]
    if is_allinone:
        if len(color) < len(features):
            import colorcet as cc
            import seaborn as sns
            color = sns.color_palette(cc.glasbey, n_colors=len(features)).as_hex()
        _, ax = plt.subplots(1,1)
        for idx, feature in enumerate(features):
            y=[ i for i in list(mtx.loc[feature, :])]
            sm_x, sm_y = sm_lowess(y, x,  frac=1./5., it=5, return_sorted = True).T
            ax.plot(sm_x, sm_y, color=color[idx], label=feature, zorder=5)
            #ax.plot(x, y, 'k.')
        if is_legend:
            ax.legend()

    else:
        for feature in features:
            _, ax = plt.subplots(1,1)
            y=[ i for i in list(mtx.loc[feature, :])]
            sm_x, sm_y = sm_lowess(y, x,  frac=1./5., it=5, return_sorted = True).T
            ax.plot(sm_x, sm_y, color='red', label=feature, zorder=5)
            ax.plot(x, y, 'k.', zorder=0)
            ax.set_title(feature)
            if is_legend:
                ax.legend()
#endf marker_line_pesudo


def regulator_dot_correlation(tfadata, branch, dot_size=14, ax=None):
    """
    Selected regulator shown in dot plot ordered by the correlation along a trajectory branch

    Parameters
    ----------
    tfadata : AnnData
        Annotated data of transcription factors
    branch : str
        branch name
    dot_size : int
        dot size
    ax : matplotlib.axes.Axes
        axes to plot


    Returns
    -------
    None
    """
    from adjustText import adjust_text

    if f"regulator_df_{branch}" not in tfadata.uns.keys():
        raise ValueError(f"regulator_df_{branch} not in tfadata.uns.keys()\n please run tl.branch_regulator_detect first!")

    ax = ax or plt.gca()

    b_correlation_df = tfadata.uns[f"regulator_df_{branch}"]

    sns.scatterplot(x=range(0, b_correlation_df.shape[0]), y = b_correlation_df.loc[:, "score"], ax=ax)

    texts = []
    for idx, (i,r) in enumerate(b_correlation_df.iterrows()):
        texts.append(ax.text(x=list(range(0, b_correlation_df.shape[0]))[idx], y=list(b_correlation_df.loc[:, "score"])[idx],s=i, size=dot_size))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    ax.set_xlabel(f"{branch} TFs")
    ax.set_ylabel(f"{branch}_correlation")
#endf regulator_dot_correlation



def regulator_heatmap(adata, tfadata, branch, figsize=(20,13), **args):
    """
    Plot heatmap of regulators calculated by phlower

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix
    tfadata: AnnData
        Annotated data matrix of transcription factors
    branch: str
        branch name
    figsize: tuple
        figure size
    args: dict
        additional arguments for sns.heatmap

    Returns
    -------
    fig, axes: tuple
        figure and axes
    """


    if f"regulator_df_{branch}" not in tfadata.uns.keys() or \
         f"regulator_tf_mat_{branch}" not in tfadata.uns.keys() or \
         f"regulator_gene_mat_{branch}" not in tfadata.uns.keys():
             raise Exception(f"Regulators have not been calculated for branch {branch}\n please run tl.branch_regulator_detect first!")

    b_correlation_df = tfadata.uns[f"regulator_df_{branch}"]
    d_tf2gene = TF_to_genes(list(tfadata.uns[f"regulator_tf_mat_{branch}"].index), ones=False)

    mat_tf = branch_heatmap_matrix(tfadata.uns[f"regulator_tf_mat_{branch}"].loc[b_correlation_df.index, :], max_features=100, var_cutoff=0.6)
    gene_idx = [d_tf2gene[i] for i in mat_tf.index]
    mat_gene = branch_heatmap_matrix(tfadata.uns[f"regulator_gene_mat_{branch}"], label_markers=gene_idx)
    fig, axes = plt.subplots(1,2, figsize=figsize)
    sns.heatmap(mat_tf, ax=axes[0],  cbar_kws = dict(use_gridspec=False,location="bottom"), cmap=plt.cm.inferno_r, **args)
    sns.heatmap(mat_gene, ax=axes[1],  cbar_kws = dict(use_gridspec=False,location="bottom"), cmap=plt.cm.inferno_r, **args)
    axes[0].set_title(f"{branch} TFs")
    axes[0].set_yticks(np.arange(mat_tf.shape[0])+ 0.5, mat_tf.index, fontsize="10", rotation=0)
    axes[1].set_title(f"{branch} Genes")
    axes[1].set_yticks(np.arange(mat_gene.shape[0])+ 0.5, mat_gene.index, fontsize="10", rotation=0)
    fig.show()
    return fig,axes
#endf regulator_heatmap

