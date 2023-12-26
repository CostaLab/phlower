import numpy as np
import pandas as pd
import networkx as nx
from anndata import AnnData
from datetime import datetime
from typing import Union, Optional, Tuple, Sequence, Dict, Any


def stream_tree_order_end_nodes(stree):
    """
    construct a dataframe to list all branches with 2 milestones ends
    like:
    	from	from_id	to	to_id	length	directed
        0	root	-1	0_28	0	1	True
        1	0_28	0	1_77	1	1

    Parameters
    ----------
    stree: `nx.DiGraph`
        the stream tree
    """
    #return [a,b if() for a,b in node_pair_list]

    from_list = []
    fromid_list = []
    toid_list = []
    to_list = []
    len_list = []
    idx_list = []

    for idx, (a,b) in enumerate(stree.edges()):
        i1 = a.split("_")
        i2 = b.split("_")
        ## pairs with root
        if i1[0] == 'root':
            from_list.append(a)
            fromid_list.append(-1)
            to_list.append(b)
            toid_list.append(i2[0])
            len_list.append(int(i2[1]) + 1)
            idx_list.append(idx)
        elif i2[0] == "root":
            from_list.append(b)
            fromid_list.append(-1)
            to_list.append(a)
            toid_list.append(i1[0])
            len_list.append(int(i1[1]) + 1)
            idx_list.append(idx)
        elif int(i1[0]) > int(i2[0]):
            from_list.append(b)
            fromid_list.append(i2[0])
            to_list.append(a)
            toid_list.append(i1[0])
            len_list.append(int(i1[1]) - int(i2[1]) + 1)
            idx_list.append(idx)
        else:
            from_list.append(a)
            fromid_list.append(i1[0])
            to_list.append(b)
            toid_list.append(i2[0])
            len_list.append(int(i2[1]) - int(i1[1]) + 1)
            idx_list.append(idx)
    df = pd.DataFrame({
        "from": from_list,
        "from_id": map(int, fromid_list),
        "to": to_list,
        "to_id": map(int, toid_list),
        #"length": len_list,
        "length": 1,
        "directed": [True]*len(len_list),
    })

    df.sort_values(by=['from_id', 'to_id'], inplace=True)
    #df.index = range(0, max(df.index))
    df.reset_index(inplace=True, drop=True)

    return df
#endf stream_tree_order_end_nodes


def pnt2line(pnt, start, end):
    """
    http://www.fundza.com/vectors/point2line/index.html
    calulate the distance from a point to a segment(start,end)

    Parameters
    ----------
    pnt: 1d array
    start: 1d array
    end: 1d array

    Returns
    -------
    distance: float
        euclidean distance from pnt to the segment
    nearest: 1d array
        the nearest point on the segment
    """
    pnt = np.array(pnt)
    start = np.array(start)
    end = np.array(end)

    line_vec = end - start
    pnt_vec = pnt - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec/line_len
    pnt_vec_scaled = pnt_vec/line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)
    #print(t)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = line_vec * t
    dist = np.linalg.norm(nearest -  pnt_vec)
    nearest = nearest + start
    return (dist, nearest)

def tree_milestone_belonging(adata, stream_tree='stream_tree', df_branches=None, cumsum_name='cumsum_mean', iscopy=False, verbose=True):
    """
    calculate the belonging of each node, top two nearest milestone.
    each node calculate all milestone distances.
    for the top 2, which top are in the branches list

    Should be similar as stream, each node need project to the segment of each branch to decide which branch it belongs to.
    otherwise there would be a problem of branch belong assigning, e.g.
    The following x is close to o2 and o3, but x is in the branch o1-o2.

             |-----------o
             |         |-o3
    ---------o1-----x--o2
             |         |-----o
             |------------------------

    Parameters
    ----------
    adata: AnnData
        adata that contains the stream tree
    stream_tree: str
        key of the stream tree in adata.uns
    df_branches: pd.DataFrame
        dataframe to store branches structure
    iscopy: bool
        return a copy of adata if True
    verbose: bool
        print verbose information
    """
    #stream_tree = 'stream_tree'
    #df_branches = df
    cumsum_mean = adata.obsm[cumsum_name]
    adata = adata.copy() if iscopy else adata

    ## create a dataframe to store cumsum_dist
    df_dist = pd.DataFrame(columns=  [f"{df_branches.loc[j, 'from']}->{df_branches.loc[j, 'to']}" for j in df_branches.index])
    cumsum_branch = []
    df_percent = pd.DataFrame(columns=  ['start_milestone', 'end_milestone', 'start_pct', 'end_pct', 'nearest'])
    for i in range(adata.n_obs): ## each node
        #print(i)
        dist_dic = {}
        shortest_dist = -10 ## intialized as a negative number
        shortest_info = []
        for j in df_branches.index:
            start_name = df_branches.loc[j, 'from']
            end_name = df_branches.loc[j, 'to']
            start = nx.get_node_attributes(adata.uns[stream_tree], 'cumsum')[start_name]
            end = nx.get_node_attributes(adata.uns[stream_tree], 'cumsum')[end_name]
            pnt = cumsum_mean[i, :]
            dist, nearest = pnt2line(pnt, start, end)
            dstart = np.linalg.norm(nearest-start)
            dend = np.linalg.norm(nearest-end)
            #print(f'[{start_name}  ->  {end_name}]   ', round(dist,2), "    starts: ", round( dstart,2), "    end: ",round(dend, 2))
            dist_dic[f'{start_name}->{end_name}'] = [dist]

            #store the shortest dist info
            if (shortest_dist < 0) or (shortest_dist > dist):
                shortest_dist = dist
                shortest_info = (start_name, end_name, 1-(dstart/(dstart+dend)), 1-(dend/(dstart+dend)), nearest)

        #print(shortest_info)
        df_percent = pd.concat([df_percent,
                                pd.DataFrame({
                                    'start_milestone':[shortest_info[0]],
                                    'end_milestone':[shortest_info[1]],
                                    'start_pct':[shortest_info[2]],
                                    'end_pct':[shortest_info[3]],
                                    'nearest':[shortest_info[4]]
                                })])
        cumsum_branch.append(f'{shortest_info[0]}->{shortest_info[1]}')
        df_dist = pd.concat([df_dist, pd.DataFrame(dist_dic)])
        #break

    if verbose:
        print(datetime.now(), "adding cells cumsum branch assignment[cumsum_branch] to adata.obs...")
    adata.obs['cumsum_branch'] = cumsum_branch

    if verbose:
        print(datetime.now(), "adding cells cumsum dist [cumsum_dist] to adata.obsm...")
    df_dist.index = adata.obs_names
    adata.obsm['cumsum_dist'] = df_dist

    if verbose:
        print(datetime.now(), "adding cells percentage to milestones [cumsum_percent] to adata.obsm...")

    df_percent.index = adata.obs_names
    adata.obsm['cumsum_percent'] = df_percent

    return adata if iscopy else None


def get_milestone_percentage(adata, obsm_key='cumsum_percent'):
    """
    get dataframe for dynverse like:
    	cell_id	milestone_id	percentage
           0	root	0.824578
           1	root	0.94362

    Parameters
    ----------
    adata: AnnData
        adata that contains the stream tree
    obsm_key: str
        key of the obsm that contains the percentage information
    """
    df = adata.obsm[obsm_key]
    df1 = df.loc[:, ('start_milestone', 'start_pct')]
    df2 = df.loc[:, ('end_milestone', 'end_pct')]
    df1['cell_id'] = df1.index
    df2['cell_id'] = df2.index
    df1 = df1.rename(columns={"start_milestone": "milestone_id", "start_pct": "percentage"})
    df2 = df2.rename(columns={"end_milestone": "milestone_id", "end_pct": "percentage"})

    return pd.concat([df1, df2])
