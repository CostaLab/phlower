import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, List

def plot_rank_gene_group(adata, name='markers_1_21_vs_0_17.2_21', n_genes=10, **kwargs):
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
            text_size=8,
            ):
    """
    Parameters
    ----------
    is_adjust_text: bool
        whether to adjust text to avoid overlap, very slow
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
    ax.scatter(x=df[log2fc_column],y=df[pval_column].apply(lambda x:nplog10_(x)),s=1,label="Not significant", color=not_sig_color)

    # highlight down- or up- regulated genes
    down = df[(df[log2fc_column]<=log2fc_threshold_neg)&(df[pval_column]<=pval_threshold)]
    up = df[(df[log2fc_column]>=log2fc_threshold_posi)&(df[pval_column]<=pval_threshold)]

    ax.scatter(x=down[log2fc_column],y=down[pval_column].apply(lambda x:nplog10_(x)),s=3,label="Down-regulated",color=down_color)
    ax.scatter(x=up[log2fc_column],y=up[pval_column].apply(lambda x:nplog10_(x)),s=3,label="Up-regulated",color=up_color)

    texts = []
    if genes_to_highlight is not None:
        genes_to_highlight = set(genes_to_highlight)
        texts = [plt.text(df.loc[gene,log2fc_column], nplog10_(df.loc[gene,pval_column]), '%s' %gene) for gene in genes_to_highlight if gene in df.index]
        for gene in genes_to_highlight:
            if gene in df.index:
                ax.scatter(x=df.loc[gene,log2fc_column],
                           y=nplog10_(df.loc[gene,pval_column]),
                           s=10,
                           label=gene,
                           size=text_size,
                           color='black')
            else:
                print(f'Gene {gene} not found in the dataframe.')
    else:
        for i,r in up.iterrows():
           texts.append(ax.text(x=r[log2fc_column],y=nplog10_(r[pval_column]),s=i, size=text_size))
        for i,r in down.iterrows():
           texts.append(ax.text(x=r[log2fc_column],y=nplog10_(r[pval_column]),s=i, size=text_size))


    if is_adjust_text and len(texts)>0:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))


    ax.set_xlabel(f"{log2fc_column}")
    ax.set_ylabel(f"-log10 {pval_column}")
    ax.axvline(log2fc_threshold_neg,color="grey",linestyle="--")
    ax.axvline(log2fc_threshold_posi,color="grey",linestyle="--")
    ax.axhline(nplog10_(pval_threshold),color="grey",linestyle="--")
    if show_legend:
        ax.legend()
