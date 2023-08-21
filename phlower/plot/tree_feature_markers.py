import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            log2fc_threshold=2,
            pval_threshold=0.01,
            show_legend=True,
            up_color='red',
            down_color='blue',
            not_sig_color='grey',
            ax=None,
            ):
    """

    """
    if ax is None:
        ax = plt.gca()

    df.index = df[gene_column]
    ax.scatter(x=df[log2fc_column],y=df[pval_column].apply(lambda x:-np.log10(x)),s=1,label="Not significant", color=not_sig_color)

    # highlight down- or up- regulated genes
    down = df[(df[log2fc_column]<=-log2fc_threshold)&(df[pval_column]<=pval_threshold)]
    up = df[(df[log2fc_column]>=log2fc_threshold)&(df[pval_column]<=pval_threshold)]

    ax.scatter(x=down[log2fc_column],y=down[pval_column].apply(lambda x:-np.log10(x)),s=3,label="Down-regulated",color=down_color)
    ax.scatter(x=up[log2fc_column],y=up[pval_column].apply(lambda x:-np.log10(x)),s=3,label="Up-regulated",color=up_color)

    if genes_to_highlight is not None:
        genes_to_highlight = set(genes_to_highlight)
        for gene in genes_to_highlight:
            if gene in df.index:
                ax.scatter(x=df.loc[gene,log2fc_column],y=-np.log10(df.loc[gene,pval_column]),s=10,label=gene,color='black')
            else:
                print(f'Gene {gene} not found in the dataframe.')
    else:
        for i,r in up.iterrows():
            plt.text(x=r[log2fc_column],y=-np.log10(r[pval_column]),s=i)
        for i,r in down.iterrows():
            plt.text(x=r[log2fc_column],y=-np.log10(r[pval_column]),s=i)


    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10FDR")
    ax.axvline(-2,color="grey",linestyle="--")
    ax.axvline(2,color="grey",linestyle="--")
    ax.axhline(2,color="grey",linestyle="--")
    if show_legend:
        ax.legend()
