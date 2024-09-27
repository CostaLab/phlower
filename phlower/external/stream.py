## code adjusted from from https://github.com/pinellolab/STREAM  d20cc1faea58df10c53ee72447a9443f4b6c8e03
import os
import copy
import scipy
import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import colorcet as cc
from copy import deepcopy
from decimal import Decimal
from scipy.stats import spearmanr,mannwhitneyu,gaussian_kde,kruskal
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from pandas.api.types import is_string_dtype,is_numeric_dtype
from .stream_extra import *
from .scikit_posthocs import posthoc_conover

##TODO: use kde information to merge the buckets
##TODO: if duplicated nodes is an issue, we can calculate center of each vertex, if set the node to the nearest vertex.
##TODO: increase the size of stream_sc size based on the score.

def tqdm_show(iterable, show=True, threshold=10, **kargs):
    """
    if show use a threshold
    """
    from tqdm import tqdm
    if len(list(iterable)) < threshold:
        return iterable
    elif show:
        return tqdm(iterable)
    return tqdm(iterable)
#endf tqdm_show

def tree_label_dict(adata,
                    tree = "stream_tree",
                    from_ = "label",
                    to_ = 'original',
                    branch_label = False,
                    ):
    htree = adata.uns[tree]

    if from_  != "node_name":
        d1= nx.get_node_attributes(adata.uns[tree], from_)
    else:
        d1 = {i:i for i in adata.uns[tree].nodes()}
    #d1= nx.get_node_attributes(adata.uns[tree], from_)


    if to_ == "original":
        d2 = nx.get_node_attributes(adata.uns[tree], to_)
    elif to_ == "node_name":
        d2 = {i:i for i in adata.uns[tree].nodes()}
    #print(d2)

    if branch_label:
        dd = {v:d2[k] for k,v in d1.items()}
    else: ## only keep leave annotation
        dd = {v:d2[k][0] if len(d2[k]) == 1 else ""  for k,v in d1.items()}
    return dd

def tree_label_convert(adata, tree, from_, to_, from_attr_list, verbose=False):
    """
    when tree has several label attributes
    convert from_ to to_
    for phlower stream tree, the original is the tuple type attr, for convenience, can ignore tuples when pass parameters
    """
    if from_ == "node_name":
        d1 = {i:i for i in adata.uns[tree].nodes()}
    else:
        d1= nx.get_node_attributes(adata.uns[tree], from_)

    if verbose:
        print("from_attr_list", from_attr_list)
    if from_ == "original":
        if all(isinstance(x, tuple) for x in from_attr_list):
            pass
        else: ## cover the case you don't have to use tuple for original
            from_attr_list = [tuple([x]) for x in from_attr_list if not isinstance(x, tuple)]


    if to_ == "node_name":
        d2 = {i:i for i in adata.uns[tree].nodes()}
    else:
        d2= nx.get_node_attributes(adata.uns[tree], to_)
    if verbose:
        print("d1", d1)
        print("d2", d2)

    d3 = {v:d2[k] for k,v in d1.items()}
    if verbose:
        print("d3", d3 )
        print("from_attr_list", from_attr_list)
    ret_list = [d3[i] for i in from_attr_list]

    return ret_list
#endf

def assign_root(adata,
                root='root',
                tree='stream_tree'
                ):
    return adata.uns[tree].nodes[root]['label']


def plot_stream_sc(adata,root='root',color=None,dist_scale=1,dist_pctl=95,preference=None,
                   fig_size=(7,4.5),fig_legend_ncol=1,fig_legend_order = None,
                   vmin=None,vmax=None,alpha=0.8,
                   pad=1.08,w_pad=None,h_pad=None,
                   show_text=True,show_graph=True,
                   show_legend=True,
                   titles = None,
                   text_attr = 'original',
                   save_fig=False,fig_path=None,fig_format='pdf',
                   return_fig = False,
                   s = 30, ## size or range
                   order = True,
                   plotly=False,
                   cmap_continous = 'viridis',
                   ):
    """Generate stream plot at single cell level (aka, subway map plots)

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    root: `str`, optional (default: 'root'):
        The starting node, temporialy abandoned cause it can be decided automatically.
    color: `list` optional (default: None)
        Column names of observations (adata.obs.columns) or variable names(adata.var_names). A list of names to be plotted.
    dist_scale: `float`,optional (default: 1)
        Scaling factor to scale the distance from cells to tree branches
        (by default, it keeps the same distance as in original manifold)
    dist_pctl: `int`, optional (default: 95)
        Percentile of cells' distances from branches (between 0 and 100) used for calculating the distances between branches.
    preference: `list`, optional (default: None):
        The preference of nodes. The branch with speficied nodes are preferred and put on the top part of stream plot.
        The higher ranks the node have, the closer to the top the branch with that node is.
    fig_size: `tuple`, optional (default: (7,4.5))
        figure size.
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.Only valid for ategorical variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values. If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots, as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots, as a fraction of the font size. Defaults to pad.
    show_text: `bool`, optional (default: False)
        If True, node state label will be shown
    show_graph: `bool`, optional (default: False)
        If True, the learnt principal graph will be shown
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        if save_fig is True, specify figure path. if None, adata.uns['workdir'] will be used.
    fig_format: `str`, optional (default: 'pdf')
        if save_fig is True, specify figure format.
    plotly: `bool`, optional (default: False)
        if True, plotly will be used to make interactive plots
    cmap_continous: `str`, optional (default: 'viridis')
        continuous values cmap
    Returns
    -------
    updates `adata` with the following fields.
    X_stream_root: `numpy.ndarray` (`adata.obsm['X_stream_root']`)
        Store #observations × 2 coordinates of cells in subwaymap plot.
    stream_root: `dict` (`adata.uns['stream_root']`)
        Store the coordinates of nodes ('nodes') and edges ('edges') in subwaymap plot.
    """
    #print("Minor adjusted from https://github.com/pinellolab/STREAM  d20cc1faea58df10c53ee72447a9443f4b6c8e03")
    ## 1. node_name is the node name
    ## 2. label is like S1 S2.... from stream
    ## 3. original is the name we specified

    if isinstance(preference, dict):
        if list(preference.keys())[0] == "original":
            preference = tree_label_convert(adata, "stream_tree", from_="original", to_='label', from_attr_list=preference["original"])
        elif list(preference.keys())[0] == "node_name":
            preference = tree_label_convert(adata, "stream_tree", from_="node_name", to_='label', from_attr_list=preference["node_name"])
        elif list(preference.keys())[0] == "label":
            preference = tree_label_convert(adata, "stream_tree", from_="label", to_='label', from_attr_list=preference["label"])

    #print(preference)
    root = assign_root(adata, root=root)
    figs = []


    dd = {}
    if text_attr == "original":
        dd = tree_label_dict(adata,tree = "stream_tree",from_ = "label",to_ = "original")
    elif text_attr == "node_name":
        dd = tree_label_dict(adata,tree = "stream_tree",from_ = "label",to_ = "node_name",branch_label=True )

    if(fig_path is None):
        fig_path = adata.uns['workdir']
    fig_size = mpl.rcParams['figure.figsize'] if fig_size is None else fig_size

    if(color is None):
        color = ['group']
        #color = ['label']
    ###remove duplicate keys

    if titles is not None:
        if len(titles) != len(color):
            titles = None
            print("warning: titles number is not consistent with color")


    #for acolor in color: cluster name should be string
    #    if acolor not in adata.obs.columns:
    #        pass
    #    else:
    #        print("warning: change {acolor} to type string if not!")
    #        adata.obs[acolor] = adata.obs[acolor].astype(str)

    color = list(dict.fromkeys(color))


    dict_ann = dict()
    for ann in color:
        if(ann in adata.obs.columns):
            dict_ann[ann] = adata.obs[ann]
            #print(adata.obs[ann]) ##----------------------------
        elif(ann in adata.var_names):
            dict_ann[ann] = adata.obs_vector(ann)
        else:
            raise ValueError("could not find '%s' in `adata.obs.columns` and `adata.var_names`"  % (ann))

    stream_tree = adata.uns['stream_tree']
    ft_node_label = nx.get_node_attributes(stream_tree,'label')
    label_to_node = {value: key for key,value in nx.get_node_attributes(stream_tree,'label').items()}
    if(root not in label_to_node.keys()):
        raise ValueError("There is no root '%s'" % root)

    add_stream_sc_pos(adata,root=root,dist_scale=dist_scale,dist_pctl=dist_pctl,preference=preference)
    stream_nodes = adata.uns['stream_'+root]['nodes']
    stream_edges = adata.uns['stream_'+root]['edges']

    df_plot = pd.DataFrame(index=adata.obs.index,data = adata.obsm['X_stream_'+root],
                           columns=['pseudotime','dist'])
    for ann in color:
        df_plot[ann] = dict_ann[ann]
    df_plot_shuf = df_plot.sample(frac=1,random_state=100)

    legend_order = {ann:np.unique(df_plot_shuf[ann]) for ann in color if is_string_dtype(df_plot_shuf[ann])}
    if(fig_legend_order is not None):
        if(not isinstance(fig_legend_order, dict)):
            raise TypeError("`fig_legend_order` must be a dictionary")
        for ann in fig_legend_order.keys():
            if(ann in legend_order.keys()):
                legend_order[ann] = fig_legend_order[ann]
            else:
                print("'%s' is ignored for ordering legend labels due to incorrect name or data type" % ann)

    if(plotly):
        for ann in color:
            fig = px.scatter(df_plot_shuf, x='pseudotime', y='dist',color=ann,
                                 opacity=alpha,
                                 color_continuous_scale=px.colors.sequential.Viridis,
                                 color_discrete_map=adata.uns[ann+'_color'] if ann+'_color' in adata.uns_keys() else {})
            if(show_graph):
                for edge_i in stream_edges.keys():
                    branch_i_pos = stream_edges[edge_i]
                    branch_i = pd.DataFrame(branch_i_pos,columns=range(branch_i_pos.shape[1]))
                    for ii in np.arange(start=0,stop=branch_i.shape[0],step=2):
                        if(branch_i.iloc[ii,0]==branch_i.iloc[ii+1,0]):
                            fig.add_trace(go.Scatter(x=branch_i.iloc[[ii,ii+1],0],
                                                       y=branch_i.iloc[[ii,ii+1],1],
                                                       mode='lines',
                                                       opacity=0.8,
                                                       line=dict(color='#767070', width=3),
                                                       showlegend=False))
                        else:
                            fig.add_trace(go.Scatter(x=branch_i.iloc[[ii,ii+1],0],
                                                       y=branch_i.iloc[[ii,ii+1],1],
                                                       mode='lines',
                                                       line=dict(color='black', width=3),
                                                       showlegend=False))
            if(show_text):
                fig.add_trace(go.Scatter(x=np.array(list(stream_nodes.values()))[:,0],
                                           y=np.array(list(stream_nodes.values()))[:,1],
                                           mode='text',
                                           opacity=1,
                                           marker=dict(size=1.5*mpl.rcParams['lines.markersize'],color='#767070'),
                                           text=[ft_node_label[x] if not dd else str(dd.get(ft_node_label[x],ft_node_label[x])) for x in stream_nodes.keys()],
                                           textposition="bottom center",
                                           name='states',
                                           showlegend=False),)
            fig.update_layout(legend= {'itemsizing': 'constant'},
                              xaxis={'showgrid': False,'zeroline': False,},
                              yaxis={'visible':False},
                              width=800,height=500)
            fig.show(renderer="notebook")
    else:
        for i, ann in enumerate(tqdm_show(color, desc='stream sc plotting')):
            fig = plt.figure(figsize=(fig_size[0],fig_size[1]))
            ax_i = fig.add_subplot(1,1,1)
            if(is_string_dtype(df_plot[ann])): ## plot groups
                sns_palette = sns.color_palette(cc.glasbey, n_colors=len(set(df_plot_shuf[ann]))).as_hex()
                if ann+'_color' not in adata.uns:
                    adata.uns[ann+'_color'] = {y:sns_palette[idx]  for idx, y in enumerate(set(df_plot_shuf[ann]))}
                sc_i=sns.scatterplot(ax=ax_i,
                                    x='pseudotime', y='dist',
                                    hue=ann,hue_order = legend_order[ann],
                                    data=df_plot_shuf,
                                    alpha=alpha,linewidth=0,
                                    s=s,
                                    palette= adata.uns[ann+'_color'] \
                                            if (ann+'_color' in adata.uns_keys()) and (set(adata.uns[ann+'_color'].keys()) >= set(np.unique(df_plot_shuf[ann]))) \
                                            else sns_palette
                                    )

                legend_handles, legend_labels = ax_i.get_legend_handles_labels()
                ax_i.legend(handles=legend_handles, labels=legend_labels,
                            bbox_to_anchor=(1, 0.5), loc='center left', ncol=fig_legend_ncol,
                            frameon=False,
                            )

                if not show_legend:
                    ax_i.get_legend().remove()

                ### remove legend title
                # ax_i.get_legend().texts[0].set_text("")
            else: ## plot variables
                vmin_i = df_plot[ann].min() if vmin is None else vmin
                vmax_i = df_plot[ann].max() if vmax is None else vmax
                cell_order = np.argsort(df_plot_shuf['pseudotime'], kind="stable") ## order cells, not working
                if isinstance(cmap_continous, str):
                    if  cmap_continous not in mpl.colormaps:
                        cmap_continous = "viridis"
                        print("warning: wrong cmap, use default viridis!")
                        cmap_obj = plt.get_cmap(cmap_continous)
                    else:
                        cmap_obj = plt.get_cmap(cmap_continous)
                elif isinstance(cmap_continous, mpl.colors.ListedColormap):
                    cmap_obj = cmap_continous
                else:
                    cmap_obj = plt.get_cmap("viridis")
                    print("warning: wrong cmap name or mpl.colors.ListedColormap, use default viridis!")




                if (isinstance(s, list) or isinstance(s, tuple)) and len(s)==2: ## variated size of point size range e.g. (2, 50)
                    s_min = s[0]
                    s_max = s[1]
                    df_plot_shuf['weight'] = s_max*((df_plot_shuf[ann] - min(df_plot_shuf[ann]))/(max(df_plot_shuf[ann]) - min(df_plot_shuf[ann])))
                    df_plot_shuf['weight'] = [i if i > s_min else s_min for i in df_plot_shuf['weight']]
                    if order:
                        ## use zorder to split the data get larger points over the smaller ones
                        zorder_bins = 20
                        df_plot_shuf['zorder'] = pd.cut(df_plot_shuf['weight'], bins=zorder_bins, labels=np.arange(1, zorder_bins+1))
                        for a_cut in set(df_plot_shuf['zorder']):
                            df_plot_shuf_i = df_plot_shuf[df_plot_shuf['zorder']==a_cut]
                            if df_plot_shuf_i.empty:
                                continue
                            sc_i=ax_i.scatter(df_plot_shuf_i['pseudotime'], df_plot_shuf_i['dist'],
                                              c=df_plot_shuf_i[ann],cmap=cmap_obj,
                                              vmin=vmin_i,vmax=vmax_i,
                                              alpha=alpha,linewidth=0,
                                              s=df_plot_shuf_i['weight'],
                                              zorder=int(a_cut),
                                              )

                    else:
                        sc_i = ax_i.scatter(list(df_plot_shuf['pseudotime']), list(df_plot_shuf['dist']), s=list(df_plot_shuf['weight']),
                                        c=list(df_plot_shuf[ann]),vmin=vmin_i,vmax=vmax_i,alpha=alpha, cmap = cmap_obj)#, zorder=[int(i*10) for i in df_plot_shuf['weight']])
                elif isinstance(s, int) or isinstance(s, float): ## fix point size
                    if order:
                        zorder_bins = 20
                        df_plot_shuf['zorder'] = pd.cut(df_plot_shuf[ann], bins=zorder_bins, labels=np.arange(1, zorder_bins+1))
                        for a_cut in set(df_plot_shuf['zorder']):
                            df_plot_shuf_i = df_plot_shuf[df_plot_shuf['zorder']==a_cut]
                            if df_plot_shuf_i.empty:
                                continue
                            sc_i=ax_i.scatter(df_plot_shuf_i['pseudotime'], df_plot_shuf_i['dist'],
                                              c=df_plot_shuf_i[ann],cmap=cmap_obj,
                                              vmin=vmin_i,vmax=vmax_i,
                                              alpha=alpha,linewidth=0,
                                              s=s,
                                              zorder=int(a_cut),
                                              )

                    else:
                        sc_i = ax_i.scatter(list(df_plot_shuf['pseudotime']), list(df_plot_shuf['dist']), s=s,
                                        c=list(df_plot_shuf[ann]),vmin=vmin_i,vmax=vmax_i,alpha=alpha, cmap = cmap_obj )
                if show_legend:
                    cbar = plt.colorbar(sc_i,ax=ax_i, pad=0.01, fraction=0.05, aspect=40)
                    cbar.solids.set_edgecolor("face")
                    cbar.ax.locator_params(nbins=5)
            if(show_graph):
                for edge_i in stream_edges.keys():
                    branch_i_pos = stream_edges[edge_i]
                    branch_i = pd.DataFrame(branch_i_pos,columns=range(branch_i_pos.shape[1]))
                    for ii in np.arange(start=0,stop=branch_i.shape[0],step=2):
                        if(branch_i.iloc[ii,0]==branch_i.iloc[ii+1,0]):
                            ax_i.plot(branch_i.iloc[[ii,ii+1],0],branch_i.iloc[[ii,ii+1],1],
                                      c = '#767070',alpha=0.8)
                        else:
                            ax_i.plot(branch_i.iloc[[ii,ii+1],0],branch_i.iloc[[ii,ii+1],1],
                                      c = 'black',alpha=1)
            if(show_text):
                for node_i in stream_tree.nodes():
                    ax_i.text(stream_nodes[node_i][0],stream_nodes[node_i][1],ft_node_label[node_i] if not dd else str(dd.get(ft_node_label[node_i],ft_node_label[node_i])),
                              color='black',fontsize=0.9*mpl.rcParams['font.size'],
                              weight="bold",
                               ha='left', va='bottom')
            ax_i.set_xlabel("pseudotime",labelpad=2)
            ax_i.spines['left'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.get_yaxis().set_visible(False)
            ax_i.locator_params(axis='x',nbins=8)
            ax_i.tick_params(axis="x",pad=-1)
            annots = arrowed_spines(ax_i, locations=('bottom right',),
                                    lw=ax_i.spines['bottom'].get_linewidth()*1e-5)
            title = ""
            if titles is None:
                title = ann
            else:
                title = titles[i]
            ax_i.set_title(title)
            plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
            if(save_fig):
                #file_path_S = os.path.join(fig_path,root)
                #if(not os.path.exists(file_path_S)):
                #    os.makedirs(file_path_S)
                #plt.savefig(os.path.join(file_path_S,'stream_sc_' + slugify(ann) + '.' + fig_format),pad_inches=1,bbox_inches='tight')
                plt.savefig(fig_path,pad_inches=1,bbox_inches='tight')
                #plt.close(fig)
            if return_fig:
                figs.append(fig)
            else:
                #plt.close(fig)
                pass
    if return_fig:
        return figs




def plot_stream(adata,root='root',color = None,preference=None,dist_scale=0.9,
                factor_num_win=10,factor_min_win=2.0,factor_width=2.5,factor_nrow=200,factor_ncol=400,
                log_scale = False,factor_zoomin=100.0,
                fig_size=(7,4.5),fig_legend_order=None,fig_legend_ncol=1,
                fig_colorbar_aspect=30,
                show_legend=True,
                titles=None,
                vmin=None,vmax=None,
                pad=1.08,w_pad=None,h_pad=None,
                save_fig=False,
                return_fig=False,
                fig_path=None,
                fig_format='pdf',
                cmap_continous = 'viridis',
                ):
    """Generate stream plot at density level

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    root: `str`, optional (default: 'root'):
        The starting node, temporialy abandoned cause it can be decided automatically.
    color: `list` optional (default: None)
        Column names of observations (adata.obs.columns) or variable names(adata.var_names). A list of names to be plotted.
    preference: `list`, optional (default: None):
        The preference of nodes. The branch with speficied nodes are preferred and put on the top part of stream plot.
        The higher ranks the node have, the closer to the top the branch with that node is.
    dist_scale: `float`,optional (default: 0.9)
        Scaling factor. It controls the width of STREAM plot branches. The smaller, the thinner the branch will be.
    factor_num_win: `int`, optional (default: 10)
        Number of sliding windows used for making stream plot. It controls the smoothness of STREAM plot.
    factor_min_win: `float`, optional (default: 2.0)
        The minimum number of sliding windows. It controls the resolution of STREAM plot. The window size is calculated based on shortest branch. (suggested range: 1.5~3.0)
    factor_width: `float`, optional (default: 2.5)
        The ratio between length and width of stream plot.
    factor_nrow: `int`, optional (default: 200)
        The number of rows in the array used to plot continuous values
    factor_ncol: `int`, optional (default: 400)
        The number of columns in the array used to plot continuous values
    log_scale: `bool`, optional (default: False)
        If True,the number of cells (the width) is logarithmized when drawing stream plot.
    factor_zoomin: `float`, optional (default: 100.0)
        If log_scale is True, the factor used to zoom in the thin branches
    fig_size: `tuple`, optional (default: (7,4.5))
        figure size.
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.Only valid for ategorical variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values. If None, the respective min and max of continuous values is used.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots, as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots, as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        if save_fig is True, specify figure path. if None, adata.uns['workdir'] will be used.
    fig_format: `str`, optional (default: 'pdf')
        if save_fig is True, specify figure format.
    cmap_continous: `str`, optional (default: 'viridis')
        continuous values cmap
    Returns
    -------
    None
    """
    #print("Minor adjusted from https://github.com/pinellolab/STREAM  d20cc1faea58df10c53ee72447a9443f4b6c8e03")
    root = assign_root(adata, root=root)
    figs = []

    ## if preference is a dict, convert to stream label to ordering
    if isinstance(preference, dict):
        if list(preference.keys())[0] == "original":
            preference = tree_label_convert(adata, "stream_tree", from_="original", to_='label', from_attr_list=preference["original"])
        elif list(preference.keys())[0] == "node_name":
            preference = tree_label_convert(adata, "stream_tree", from_="node_name", to_='label', from_attr_list=preference["node_name"])
        elif list(preference.keys())[0] == "label":
            preference = tree_label_convert(adata, "stream_tree", from_="label", to_='label', from_attr_list=preference["label"])

    if(fig_path is None):
        fig_path = adata.uns['workdir']
    fig_size = mpl.rcParams['figure.figsize'] if fig_size is None else fig_size

    if(color is None):
        #color = ['label']
        color = ['group']

    if titles is not None:
        if len(titles) != len(color):
            titles = None
            print("warning: titles number is not consistent with color")



    #for acolor in color: cluster name should be string
    #    if acolor not in adata.obs.columns:
    #        pass
    #    else:
    #        print("warning: change {acolor} to type string if not!")
    #        adata.obs[acolor] = adata.obs[acolor].astype(str)


    ###remove duplicate keys
    color = list(dict.fromkeys(color))

    dict_ann = dict()
    for ann in color:
        if(ann in adata.obs.columns):
            dict_ann[ann] = adata.obs[ann]
        elif(ann in adata.var_names):
            dict_ann[ann] = adata.obs_vector(ann)
        else:
            raise ValueError("could not find '%s' in `adata.obs.columns` and `adata.var_names`"  % (ann))

    stream_tree = adata.uns['stream_tree']
    ft_node_label = nx.get_node_attributes(stream_tree,'label')
    label_to_node = {value: key for key,value in nx.get_node_attributes(stream_tree,'label').items()}
    if(root not in label_to_node.keys()):
        raise ValueError("There is no root '%s'" % root)

    if(preference!=None):
        preference_nodes = [label_to_node[x] for x in preference]
    else:
        preference_nodes = None

    legend_order = {ann:np.unique(dict_ann[ann]) for ann in color if is_string_dtype(dict_ann[ann])}
    if(fig_legend_order is not None):
        if(not isinstance(fig_legend_order, dict)):
            raise TypeError("`fig_legend_order` must be a dictionary")
        for ann in fig_legend_order.keys():
            if(ann in legend_order.keys()):
                legend_order[ann] = fig_legend_order[ann]
            else:
                print("'%s' is ignored for ordering legend labels due to incorrect name or data type" % ann)

    dict_plot = dict()

    list_string_type = [k for k,v in dict_ann.items() if is_string_dtype(v)]
    if(len(list_string_type)>0):
        dict_verts,dict_extent = \
        cal_stream_polygon_string(adata,dict_ann,root=root,preference=preference,dist_scale=dist_scale,
                                  factor_num_win=factor_num_win,factor_min_win=factor_min_win,factor_width=factor_width,
                                  log_scale=log_scale,factor_zoomin=factor_zoomin)
        dict_plot['string'] = [dict_verts,dict_extent]

    list_numeric_type = [k for k,v in dict_ann.items() if is_numeric_dtype(v)]
    if(len(list_numeric_type)>0):
        verts,extent,ann_order,dict_ann_df,dict_im_array = \
        cal_stream_polygon_numeric(adata,dict_ann,root=root,preference=preference,dist_scale=dist_scale,
                                   factor_num_win=factor_num_win,factor_min_win=factor_min_win,factor_width=factor_width,
                                   factor_nrow=factor_nrow,factor_ncol=factor_ncol,
                                   log_scale=log_scale,factor_zoomin=factor_zoomin)
        dict_plot['numeric'] = [verts,extent,ann_order,dict_ann_df,dict_im_array]

    for i, ann in enumerate(tqdm_show(color, desc='stream plotting')):
        if(is_string_dtype(dict_ann[ann])):
            if(not ((ann+'_color' in adata.uns_keys()) and (set(adata.uns[ann+'_color'].keys()) >= set(np.unique(dict_ann[ann]))))):
                ### a hacky way to generate colors from seaborn
                tmp = pd.DataFrame(index=adata.obs_names,
                                   data=np.random.rand(adata.shape[0], 2))
                tmp[ann] = dict_ann[ann]
                fig = plt.figure(figsize=fig_size)
                sc_i=sns.scatterplot(x=0,y=1,hue=ann,data=tmp,linewidth=0)
                colors_sns = sc_i.get_children()[0].get_facecolors()
                plt.close(fig)
                #colors_sns_scaled = (255*colors_sns).astype(int)

                #adata.uns[ann+'_color'] = {tmp[ann][i]:'#%02x%02x%02x' % (colors_sns_scaled[i][0], colors_sns_scaled[i][1], colors_sns_scaled[i][2])
                #                           for i in np.unique(tmp[ann],return_index=True)[1]}
                sns_palette = sns.color_palette(cc.glasbey, n_colors=len(set(tmp[ann]))).as_hex()
                adata.uns[ann+'_color'] = {y:sns_palette[idx]  for idx, y in enumerate(set(tmp[ann]))}

            dict_palette = adata.uns[ann+'_color']

            verts = dict_plot['string'][0][ann]
            extent = dict_plot['string'][1][ann]
            xmin = extent['xmin']
            xmax = extent['xmax']
            ymin = extent['ymin'] - (extent['ymax'] - extent['ymin'])*0.1
            ymax = extent['ymax'] + (extent['ymax'] - extent['ymin'])*0.1

            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1,1,1)
            legend_labels = []
            for ann_i in legend_order[ann]:
                legend_labels.append(ann_i)
                verts_cell = verts[ann_i]
                polygon = Polygon(verts_cell,closed=True,color=dict_palette[ann_i],alpha=0.8,lw=0)
                ax.add_patch(polygon)
            ax.legend(legend_labels,bbox_to_anchor=(1.03, 0.5), loc='center left', ncol=fig_legend_ncol,frameon=False,
                      columnspacing=0.4,
                      borderaxespad=0.2,
                      handletextpad=0.3,)
            if not show_legend:
                ax.get_legend().remove()
        else:
            verts = dict_plot['numeric'][0]
            extent = dict_plot['numeric'][1]
            ann_order = dict_plot['numeric'][2]
            dict_ann_df = dict_plot['numeric'][3]
            dict_im_array = dict_plot['numeric'][4]
            xmin = extent['xmin']
            xmax = extent['xmax']
            ymin = extent['ymin'] - (extent['ymax'] - extent['ymin'])*0.1
            ymax = extent['ymax'] + (extent['ymax'] - extent['ymin'])*0.1

            #clip parts according to determined polygon
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(1,1,1)
            for ann_i in ann_order:
                vmin_i = dict_ann_df[ann].loc[ann_i,:].min() if vmin is None else vmin
                vmax_i = dict_ann_df[ann].loc[ann_i,:].max() if vmax is None else vmax
                if isinstance(cmap_continous, str):
                    if  cmap_continous not in mpl.colormaps:
                        cmap_continous = "viridis"
                        print("warning: wrong cmap, use default viridis!")
                        cmap_obj = plt.get_cmap(cmap_continous)
                    else:
                        cmap_obj = plt.get_cmap(cmap_continous)
                elif isinstance(cmap_continous, mpl.colors.ListedColormap):
                    cmap_obj = cmap_continous
                else:
                    cmap_obj = plt.get_cmap("viridis")
                    print("warning: wrong cmap name or mpl.colors.ListedColormap, use default viridis!")

                im = ax.imshow(dict_im_array[ann][ann_i],interpolation='bicubic',
                               extent=[xmin,xmax,ymin,ymax],vmin=vmin_i,vmax=vmax_i,aspect='auto', cmap=cmap_obj)
                verts_cell = verts[ann_i]
                clip_path = Polygon(verts_cell, facecolor='none', edgecolor='none', closed=True)
                ax.add_patch(clip_path)
                im.set_clip_path(clip_path)

                if show_legend:
                    cbar = plt.colorbar(im, ax=ax, pad=0.04, fraction=0.02, aspect=fig_colorbar_aspect)
                    cbar.ax.locator_params(nbins=5)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel("pseudotime",labelpad=2)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.locator_params(axis='x',nbins=8)
        ax.tick_params(axis="x",pad=-1)
        annots = arrowed_spines(ax, locations=('bottom right',),
                                lw=ax.spines['bottom'].get_linewidth()*1e-5)
        title=""
        if titles is None:
            title = ann
        else:
            title = titles[i]
        ax.set_title(title)
        title = None
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if(save_fig):
            #file_path_S = os.path.join(fig_path,root)
            #if(not os.path.exists(file_path_S)):
            #    os.makedirs(file_path_S)
            #plt.savefig(os.path.join(file_path_S,'stream_' + slugify(ann) + '.' + fig_format),pad_inches=1,bbox_inches='tight')
            plt.savefig(fig_path, pad_inches=1,bbox_inches='tight')
            #plt.close(fig)

        if return_fig:
            figs.append(fig)
        else:
            #plt.close(fig)
            pass
    if return_fig:
        return figs


def detect_transition_markers(adata,marker_list=None,cutoff_spearman=0.4, cutoff_logfc = 0.25, percentile_expr=95, n_jobs = 1,min_num_cells=5,use_precomputed=True, root='root',preference=None):

    """Detect transition markers along one branch.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    marker_list: `list`, optional (default: None):
        A list of candidate markers to be scanned. Instead of scanning all available genes/peaks/kmers/motifs, this will limit the scanning to a specific list of genes/peaks/kmers/motifs
        If none, all available features (genes/peaks/kmers/motifs) will be scanned.
    cutoff_spearman: `float`, optional (default: 0.4)
        Between 0 and 1. The cutoff used for Spearman's rank correlation coefficient.
    cutoff_logfc: `float`, optional (default: 0.25)
        The log2-transformed fold change cutoff between cells around start and end node.
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Between 0 and 100. Specify the percentile of marker expression greater than 0 to filter out some extreme marker expressions.
    min_num_cells: `int`, optional (default: 5)
        The minimum number of cells in which markers are expressed.
    n_jobs: `int`, optional (default: 1)
        The number of parallel jobs to run when scaling the marker expressions .
    use_precomputed: `bool`, optional (default: True)
        If True, the previously computed scaled marker expression will be used
    root: `str`, optional (default: 'root'):
        The starting node
    preference: `list`, optional (default: None):
        The preference of nodes. The branch with speficied nodes are preferred and will be dealt with first. The higher ranks the node have, The earlier the branch with that node will be analyzed.
        This will help generate the consistent results as shown in subway map and stream plot.

    Returns
    -------
    updates `adata` with the following fields.
    scaled_marker_expr: `list` (`adata.uns['scaled_marker_expr']`)
        Scaled marker expression for marker marker detection.
    transition_markers: `dict` (`adata.uns['transition_markers']`)
        Transition markers for each branch deteced by STREAM.
    """

    root = assign_root(adata, root=root)

    file_path = os.path.join(adata.uns['workdir'],'transition_markers')
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)

    if(marker_list is None):
        print('Scanning all features ...')
        marker_list = adata.var_names.tolist()
    else:
        print('Scanning the specified marker list ...')
        ###remove duplicate keys
        marker_list = list(dict.fromkeys(marker_list))
        for marker in marker_list:
            if(marker not in adata.var_names):
                raise ValueError("could not find '%s' in `adata.var_names`"  % (marker))

    flat_tree = adata.uns['stream_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    df_marker_detection = adata.obs.copy()
    df_marker_detection.rename(columns={"label": "CELL_LABEL", "branch_lam": "lam"},inplace = True)
    if(use_precomputed and ('scaled_marker_expr' in adata.uns_keys())):
        print('Importing precomputed scaled marker expression matrix ...')
        results = adata.uns['scaled_marker_expr']
        df_results = pd.DataFrame(results).T
        if(all(np.isin(marker_list,df_results.columns.tolist()))):
            input_markers_expressed = marker_list
            df_scaled_marker_expr = df_results[input_markers_expressed]
        else:
            raise ValueError("Not all markers in `marker_list` can be found in precomputed scaled marker expression matrix. Please set `use_precomputed=False`")

    else:
        input_markers = marker_list
        df_sc = pd.DataFrame(index= adata.obs_names.tolist(),
                             data = np.array(adata[:,input_markers].X if isinstance(adata.X, np.ndarray) else  adata[:,input_markers].X.todense() , dtype=object),
                             columns=input_markers)
        #exclude markers that are expressed in fewer than min_num_cells cells
        #min_num_cells = max(5,int(round(df_marker_detection.shape[0]*0.001)))
        # print('Minimum number of cells expressing markers: '+ str(min_num_cells))

        print("Filtering out markers that are expressed in less than " + str(min_num_cells) + " cells ...")
        input_markers_expressed = np.array(input_markers)[np.where((df_sc[input_markers]>0).sum(axis=0)>min_num_cells)[0]].tolist()
        df_marker_detection[input_markers_expressed] = df_sc[input_markers_expressed].copy()

        print(str(n_jobs)+' cpus are being used ...')
        params = [(df_marker_detection,x,percentile_expr) for x in input_markers_expressed]
        pool = multiprocessing.Pool(processes=n_jobs)
        results = pool.map(scale_marker_expr,params)
        pool.close()
        adata.uns['scaled_marker_expr'] = results
        df_scaled_marker_expr = pd.DataFrame(results).T

    print(str(len(input_markers_expressed)) + ' markers are being scanned ...')
    df_marker_detection[input_markers_expressed] = df_scaled_marker_expr
    #### TG (Transition markers) along each branch
    dict_tg_edges = dict()
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    root_node = dict_label_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
#     all_branches = np.unique(df_marker_detection['branch_id']).tolist()
    for edge_i in bfs_edges:
        edge_i_alias = (dict_node_state[edge_i[0]],dict_node_state[edge_i[1]])
        if edge_i in nx.get_edge_attributes(flat_tree,'id').values():
            df_cells_edge_i = deepcopy(df_marker_detection[df_marker_detection.branch_id==edge_i])
            df_cells_edge_i['lam_ordered'] = df_cells_edge_i['lam']
        else:
            df_cells_edge_i = deepcopy(df_marker_detection[df_marker_detection.branch_id==(edge_i[1],edge_i[0])])
            df_cells_edge_i['lam_ordered'] = flat_tree.edges[edge_i]['len'] - df_cells_edge_i['lam']
        df_cells_edge_i_sort = df_cells_edge_i.sort_values(['lam_ordered'])
        df_stat_pval_qval = pd.DataFrame(columns = ['stat','logfc','pval','qval'],dtype=float)
        for markername in input_markers_expressed:
            id_initial = range(0,int(df_cells_edge_i_sort.shape[0]*0.2))
            id_final = range(int(df_cells_edge_i_sort.shape[0]*0.8),int(df_cells_edge_i_sort.shape[0]*1))
            values_initial = df_cells_edge_i_sort.iloc[id_initial,:][markername]
            values_final = df_cells_edge_i_sort.iloc[id_final,:][markername]
            diff_initial_final = abs(values_final.mean() - values_initial.mean())
            if(diff_initial_final>0):
                logfc = np.log2(max(values_final.mean(),values_initial.mean())/(min(values_final.mean(),values_initial.mean())+diff_initial_final/1000.0))
            else:
                logfc = 0
            if(logfc>cutoff_logfc):
                df_stat_pval_qval.loc[markername] = np.nan
                df_stat_pval_qval.loc[markername,['stat','pval']] = spearmanr(df_cells_edge_i_sort.loc[:,markername],\
                                                                            df_cells_edge_i_sort.loc[:,'lam_ordered'])
                df_stat_pval_qval.loc[markername,'logfc'] = logfc
        if(df_stat_pval_qval.shape[0]==0):
            print('No Transition markers are detected in branch ' + edge_i_alias[0]+'_'+edge_i_alias[1])
        else:
            p_values = df_stat_pval_qval['pval']
            q_values = multipletests(p_values, method='fdr_bh')[1]
            df_stat_pval_qval['qval'] = q_values
            dict_tg_edges[edge_i_alias] = df_stat_pval_qval[(abs(df_stat_pval_qval.stat)>=cutoff_spearman)].sort_values(['qval'])
            dict_tg_edges[edge_i_alias].to_csv(os.path.join(file_path,'transition_markers_'+ edge_i_alias[0]+'_'+edge_i_alias[1] + '.tsv'),sep = '\t',index = True)
    adata.uns['transition_markers'] = dict_tg_edges




def plot_transition_markers(adata,num_markers = 15,
                            save_fig=False,fig_path=None,fig_size=None,
                            text_attr = "original",
                            ):

    if(fig_path is None):
        fig_path = os.path.join(adata.uns['workdir'],'transition_markers')
    if(not os.path.exists(fig_path)):
        os.makedirs(fig_path)

    if text_attr == "original":
        dd = tree_label_dict(adata,tree = "stream_tree",from_ = "label",to_ = 'original')

    dict_tg_edges = adata.uns['transition_markers']
    flat_tree = adata.uns['stream_tree']
    colors = sns.color_palette("Set1", n_colors=8, desat=0.8)
    for edge_i in dict_tg_edges.keys():

        df_tg_edge_i = deepcopy(dict_tg_edges[edge_i])
        df_tg_edge_i = df_tg_edge_i.iloc[:num_markers,:]

        stat = df_tg_edge_i.stat[::-1]
        qvals = df_tg_edge_i.qval[::-1]

        pos = np.arange(df_tg_edge_i.shape[0])-1
        bar_colors = np.tile(colors[4],(len(stat),1))
        id_neg = np.arange(len(stat))[np.array(stat<0)]
        bar_colors[id_neg]=colors[2]

        if(fig_size is None):
            fig_size = (12,np.ceil(0.4*len(stat)))
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1,1,1, adjustable='box')
        ax.barh(pos,stat,align='center',height=0.8,tick_label=[''],color = bar_colors)
        ax.set_xlabel('Spearman correlation coefficient')
        edge_i0 = str(dd.get(edge_i[0],edge_i[0]))
        edge_i1 = str(dd.get(edge_i[1],edge_i[1]))

        ax.set_title("branch " + edge_i0+'_'+edge_i1)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim((min(pos)-1,max(pos)+1))

        rects = ax.patches
        for i,rect in enumerate(rects):
            if(stat[i]>0):
                alignment = {'horizontalalignment': 'left', 'verticalalignment': 'center'}
                ax.text(rect.get_x()+rect.get_width()+0.01, rect.get_y() + rect.get_height()/2.0, \
                        qvals.index[i],**alignment)
                ax.text(rect.get_x()+0.01, rect.get_y()+rect.get_height()/2.0,
                        "{:.2E}".format(Decimal(str(qvals[i]))),size=0.8*mpl.rcParams['font.size'],
                        color='black',**alignment)
            else:
                alignment = {'horizontalalignment': 'right', 'verticalalignment': 'center'}
                ax.text(rect.get_x()+rect.get_width()-0.01, rect.get_y()+rect.get_height()/2.0, \
                        qvals.index[i],**alignment)
                ax.text(rect.get_x()-0.01, rect.get_y()+rect.get_height()/2.0,
                        "{:.2E}".format(Decimal(str(qvals[i]))),size=0.8*mpl.rcParams['font.size'],
                        color='w',**alignment)
        plt.tight_layout()
        if(save_fig):
            plt.savefig(os.path.join(fig_path,'transition_markers_'+ edge_i0+'_'+edge_i1+'.pdf'),\
                        pad_inches=1,bbox_inches='tight')
            plt.close(fig)




def detect_leaf_markers(adata,marker_list=None,cutoff_zscore=1.,cutoff_pvalue=1e-2,percentile_expr=95,n_jobs = 1,min_num_cells=5,
                        use_precomputed=True, root='root',preference=None):
    """Detect leaf markers for each branch.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    marker_list: `list`, optional (default: None):
        A list of candidate markers to be scanned. Instead of scanning all available genes/peaks/kmers/motifs, this will limit the scanning to a specific list of genes/peaks/kmers/motifs
        If none, all available features (genes/peaks/kmers/motifs) will be scanned.
    cutoff_zscore: `float`, optional (default: 1.5)
        The z-score cutoff used for mean values of all leaf branches.
    cutoff_pvalue: `float`, optional (default: 1e-2)
        The p value cutoff used for Kruskal-Wallis H-test and post-hoc pairwise Conover’s test.
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Between 0 and 100. Specify the percentile of marker expression greater than 0 to filter out some extreme marker expressions.
    n_jobs: `int`, optional (default: 1)
        The number of parallel jobs to run when scaling the marker expressions .
    min_num_cells: `int`, optional (default: 5)
        The minimum number of cells in which markers are expressed.
    use_precomputed: `bool`, optional (default: True)
        If True, the previously computed scaled marker expression will be used
    root: `str`, optional (default: 'S0'):
        The starting node
    preference: `list`, optional (default: None):
        The preference of nodes. The branch with speficied nodes are preferred and will be dealt with first. The higher ranks the node have, The earlier the branch with that node will be analyzed.
        This will help generate the consistent results as shown in subway map and stream plot.

    Returns
    -------
    updates `adata` with the following fields.
    scaled_marker_expr: `list` (`adata.uns['scaled_marker_expr']`)
        Scaled marker expression for marker marker detection.
    leaf_markers_all: `pandas.core.frame.DataFrame` (`adata.uns['leaf_markers_all']`)
        All the leaf markers from all leaf branches.
    leaf_markers: `dict` (`adata.uns['leaf_markers']`)
        Leaf markers for each branch.
    """

    file_path = os.path.join(adata.uns['workdir'],'leaf_markers')
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)


    root = assign_root(adata, root=root)

    if(marker_list is None):
        print('Scanning all features ...')
        marker_list = adata.var_names.tolist()
    else:
        print('Scanning the specified marker list ...')
        ###remove duplicate keys
        marker_list = list(dict.fromkeys(marker_list))
        for marker in marker_list:
            if(marker not in adata.var_names):
                raise ValueError("could not find '%s' in `adata.var_names`"  % (marker))

    flat_tree = adata.uns['stream_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    df_marker_detection = adata.obs.copy()
    df_marker_detection.rename(columns={"label": "CELL_LABEL", "branch_lam": "lam"},inplace = True)

    if(use_precomputed and ('scaled_marker_expr' in adata.uns_keys())):
        print('Importing precomputed scaled marker expression matrix ...')
        results = adata.uns['scaled_marker_expr']
        df_results = pd.DataFrame(results).T
        if(all(np.isin(marker_list,df_results.columns.tolist()))):
            input_markers_expressed = marker_list
            df_scaled_marker_expr = df_results[input_markers_expressed]
        else:
            raise ValueError("Not all markers in `marker_list` can be found in precomputed scaled marker expression matrix. Please set `use_precomputed=False`")
    else:
        input_markers = marker_list
        df_sc = pd.DataFrame(index= adata.obs_names.tolist(),
                             data = np.array(adata[:,input_markers].X if isinstance(adata.X, np.ndarray) else  adata[:,input_markers].X.todense() , dtype=object),
                             columns=input_markers)


        #exclude markers that are expressed in fewer than min_num_cells cells
        print("Filtering out markers that are expressed in less than " + str(min_num_cells) + " cells ...")
        input_markers_expressed = np.array(input_markers)[np.where((df_sc[input_markers]>0).sum(axis=0)>min_num_cells)[0]].tolist()
        df_marker_detection[input_markers_expressed] = df_sc[input_markers_expressed].copy()

        print(str(n_jobs)+' cpus are being used ...')
        params = [(df_marker_detection,x,percentile_expr) for x in input_markers_expressed]
        pool = multiprocessing.Pool(processes=n_jobs)
        results = pool.map(scale_marker_expr,params)
        pool.close()
        adata.uns['scaled_marker_expr'] = results
        df_scaled_marker_expr = pd.DataFrame(results).T

    print(str(len(input_markers_expressed)) + ' markers are being scanned ...')
    df_marker_detection[input_markers_expressed] = df_scaled_marker_expr

    #### find marker markers that are specific to one leaf branch
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}
    if(preference!=None):
        preference_nodes = [dict_label_node[x] for x in preference]
    else:
        preference_nodes = None
    root_node = dict_label_node[root]
    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=preference_nodes)
    leaves = [k for k,v in flat_tree.degree() if v==1]
    leaf_edges = [x for x in bfs_edges if (x[0] in leaves) or (x[1] in leaves)]

    df_marker_detection['bfs_edges'] = df_marker_detection['branch_id']
    df_marker_detection.astype('object')
    for x in df_marker_detection['branch_id'].unique():
        id_ = df_marker_detection[df_marker_detection['branch_id']==x].index
        if x not in bfs_edges:
            df_marker_detection.loc[id_,'bfs_edges'] =pd.Series(index=id_,data=[(x[1],x[0])]*len(id_))

    df_leaf_markers = pd.DataFrame(columns=['zscore','H_statistic','H_pvalue']+leaf_edges)
    for marker in input_markers_expressed:
        meann_values = df_marker_detection[['bfs_edges',marker]].groupby(by = 'bfs_edges')[marker].mean()
        br_values = df_marker_detection[['bfs_edges',marker]].groupby(by = 'bfs_edges')[marker].apply(list)
        leaf_mean_values = meann_values[leaf_edges]
        leaf_mean_values.sort_values(inplace=True)
        leaf_br_values = br_values[leaf_edges]
        if(leaf_mean_values.shape[0]<=2):
            print('There are not enough leaf branches')
        else:
            zscores = scipy.stats.zscore(leaf_mean_values)
            if(abs(zscores)[abs(zscores)>cutoff_zscore].shape[0]>=1):
                if(any(zscores>cutoff_zscore)):
                    cand_br = leaf_mean_values.index[-1]
                    cand_zscore = zscores[-1]
                else:
                    cand_br = leaf_mean_values.index[0]
                    cand_zscore = zscores[0]
                list_br_values = [leaf_br_values[x] for x in leaf_edges]
                kurskal_statistic,kurskal_pvalue = kruskal(*list_br_values)
                if(kurskal_pvalue<cutoff_pvalue):
                    df_conover_pvalues= posthoc_conover(df_marker_detection[[x in leaf_edges for x in df_marker_detection['bfs_edges']]],
                                                       val_col=marker, group_col='bfs_edges', p_adjust = 'fdr_bh')
                    cand_conover_pvalues = df_conover_pvalues[~df_conover_pvalues.columns.isin([cand_br])][cand_br]
                    if(all(cand_conover_pvalues < cutoff_pvalue)):
                        df_leaf_markers.loc[marker,:] = 1.0
                        df_leaf_markers.loc[marker,['zscore','H_statistic','H_pvalue']] = [cand_zscore,kurskal_statistic,kurskal_pvalue]
                        df_leaf_markers.loc[marker,cand_conover_pvalues.index] = cand_conover_pvalues
    df_leaf_markers.rename(columns={x:dict_node_state[x[0]]+dict_node_state[x[1]]+'_pvalue' for x in leaf_edges},inplace=True)
    df_leaf_markers.sort_values(by=['H_pvalue','zscore'],ascending=[True,False],inplace=True)
    df_leaf_markers.to_csv(os.path.join(file_path,'leaf_markers.tsv'),sep = '\t',index = True)
    dict_leaf_markers = dict()
    for x in leaf_edges:
        x_alias = (dict_node_state[x[0]],dict_node_state[x[1]])
        dict_leaf_markers[x_alias] = df_leaf_markers[df_leaf_markers[x_alias[0]+x_alias[1]+'_pvalue']==1.0]
        dict_leaf_markers[x_alias].to_csv(os.path.join(file_path,'leaf_markers'+x_alias[0]+'_'+x_alias[1] + '.tsv'),sep = '\t',index = True)
    adata.uns['leaf_markers_all'] = df_leaf_markers
    adata.uns['leaf_markers'] = dict_leaf_markers

