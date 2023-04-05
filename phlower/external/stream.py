## code adjusted from from https://github.com/pinellolab/STREAM  d20cc1faea58df10c53ee72447a9443f4b6c8e03
import copy
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib as mpl
import colorcet as cc
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from pandas.api.types import is_string_dtype,is_numeric_dtype
from .stream_extra import *

##TODO: use kde information to merge the buckets
##TODO: if duplicated nodes is an issue, we can calculate center of each vertex, if set the node to the nearest vertex.


def tree_label_dict(adata,
                    tree = "stream",
                    from_ = "label",
                    to_ = 'original'
                    ):
    htree = adata.uns[tree]
    d1= nx.get_node_attributes(adata.uns[tree], from_)
    d2 = nx.get_node_attributes(adata.uns[tree], to_)

    dd = {v:d2[k] for k,v in d1.items()}
    return dd



def assign_root(adata,
                root='root',
                tree='stream_tree'
                ):
    return adata.uns[tree].nodes['root']['label']


def plot_stream_sc(adata,root='S0',color=None,dist_scale=1,dist_pctl=95,preference=None,
                   fig_size=(7,4.5),fig_legend_ncol=1,fig_legend_order = None,
                   vmin=None,vmax=None,alpha=0.8,
                   pad=1.08,w_pad=None,h_pad=None,
                   show_text=True,show_graph=True,
                   text_attr = 'original',
                   save_fig=False,fig_path=None,fig_format='pdf',
                   plotly=False):
    """Generate stream plot at single cell level (aka, subway map plots)

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    root: `str`, optional (default: 'S0'):
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
    Returns
    -------
    updates `adata` with the following fields.
    X_stream_root: `numpy.ndarray` (`adata.obsm['X_stream_root']`)
        Store #observations × 2 coordinates of cells in subwaymap plot.
    stream_root: `dict` (`adata.uns['stream_root']`)
        Store the coordinates of nodes ('nodes') and edges ('edges') in subwaymap plot.
    """
    print("Minor adjusted from https://github.com/pinellolab/STREAM  d20cc1faea58df10c53ee72447a9443f4b6c8e03")

    root = assign_root(adata)

    dd = {}
    if text_attr == "original":
        dd = tree_label_dict(adata,tree = "stream_tree",from_ = "label",to_ = 'original')


    if(fig_path is None):
        fig_path = adata.uns['workdir']
    fig_size = mpl.rcParams['figure.figsize'] if fig_size is None else fig_size

    if(color is None):
        color = ['group']
        #color = ['label']
    ###remove duplicate keys

    for acolor in color:
        if acolor not in adata.obs.columns:
            pass
        else:
            print("warning: change {acolor} to type string if not!")
            adata.obs[acolor] = adata.obs[acolor].astype(str)

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
        for i,ann in enumerate(color):
            fig = plt.figure(figsize=(fig_size[0],fig_size[1]))
            ax_i = fig.add_subplot(1,1,1)
            if(is_string_dtype(df_plot[ann])):
                sns_palette = sns.color_palette(cc.glasbey, n_colors=len(set(df_plot_shuf[ann]))).as_hex()
                adata.uns[ann+'_color'] = {y:sns_palette[idx]  for idx, y in enumerate(set(df_plot_shuf[ann]))}
                sc_i=sns.scatterplot(ax=ax_i,
                                    x='pseudotime', y='dist',
                                    hue=ann,hue_order = legend_order[ann],
                                    data=df_plot_shuf,
                                    alpha=alpha,linewidth=0,
                                    palette= adata.uns[ann+'_color'] \
                                            if (ann+'_color' in adata.uns_keys()) and (set(adata.uns[ann+'_color'].keys()) >= set(np.unique(df_plot_shuf[ann]))) \
                                            else sns_palette
                                    )
                legend_handles, legend_labels = ax_i.get_legend_handles_labels()
                ax_i.legend(handles=legend_handles, labels=legend_labels,
                            bbox_to_anchor=(1, 0.5), loc='center left', ncol=fig_legend_ncol,
                            frameon=False,
                            )

                ### remove legend title
                # ax_i.get_legend().texts[0].set_text("")
            else:
                vmin_i = df_plot[ann].min() if vmin is None else vmin
                vmax_i = df_plot[ann].max() if vmax is None else vmax
                sc_i = ax_i.scatter(df_plot_shuf['pseudotime'], df_plot_shuf['dist'],
                                    c=df_plot_shuf[ann],vmin=vmin_i,vmax=vmax_i,alpha=alpha)
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
            ax_i.set_title(ann)
            plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
            if(save_fig):
                file_path_S = os.path.join(fig_path,root)
                if(not os.path.exists(file_path_S)):
                    os.makedirs(file_path_S)
                plt.savefig(os.path.join(file_path_S,'stream_sc_' + slugify(ann) + '.' + fig_format),pad_inches=1,bbox_inches='tight')
                plt.close(fig)




def plot_stream(adata,root='S0',color = None,preference=None,dist_scale=0.9,
                factor_num_win=10,factor_min_win=2.0,factor_width=2.5,factor_nrow=200,factor_ncol=400,
                log_scale = False,factor_zoomin=100.0,
                fig_size=(7,4.5),fig_legend_order=None,fig_legend_ncol=1,
                fig_colorbar_aspect=30,
                vmin=None,vmax=None,
                pad=1.08,w_pad=None,h_pad=None,
                save_fig=False,fig_path=None,fig_format='pdf'):
    """Generate stream plot at density level

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    root: `str`, optional (default: 'S0'):
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
    Returns
    -------
    None
    """
    print("Minor adjusted from https://github.com/pinellolab/STREAM  d20cc1faea58df10c53ee72447a9443f4b6c8e03")
    root = assign_root(adata)

    if(fig_path is None):
        fig_path = adata.uns['workdir']
    fig_size = mpl.rcParams['figure.figsize'] if fig_size is None else fig_size

    if(color is None):
        #color = ['label']
        color = ['group']

    for acolor in color:
        if acolor not in adata.obs.columns:
            pass
        else:
            print("warning: change {acolor} to type string if not!")
            adata.obs[acolor] = adata.obs[acolor].astype(str)


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

    for ann in color:
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
                im = ax.imshow(dict_im_array[ann][ann_i],interpolation='bicubic',
                               extent=[xmin,xmax,ymin,ymax],vmin=vmin_i,vmax=vmax_i,aspect='auto')
                verts_cell = verts[ann_i]
                clip_path = Polygon(verts_cell, facecolor='none', edgecolor='none', closed=True)
                ax.add_patch(clip_path)
                im.set_clip_path(clip_path)
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
        ax.set_title(ann)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if(save_fig):
            file_path_S = os.path.join(fig_path,root)
            if(not os.path.exists(file_path_S)):
                os.makedirs(file_path_S)
            plt.savefig(os.path.join(file_path_S,'stream_' + slugify(ann) + '.' + fig_format),pad_inches=1,bbox_inches='tight')
            plt.close(fig)
