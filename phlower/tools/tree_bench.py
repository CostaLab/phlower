import numpy as np
import pandas as pd
import networkx as nx
from anndata import AnnData
from datetime import datetime
from typing import Union, Optional, Tuple, Sequence, Dict, Any
from sklearn.neighbors import NearestNeighbors
from .cumsum_utils import node_cumsum_mean

def a2b_data_index(adata, bdata, indices):
    """
    adata is a subset of bdata
    given adata indices, return bdata indices
    """
    #indices = sorted(indices)
    bc = adata.obs_names.take(indices)
    #np.where(np.in1d([str(i) for i in bdata.obs_names], bc))[0]
    bidxs = [list(bdata.obs_names).index(x) for x in bc]
    return {aidx:bidx for aidx,bidx in zip(indices, bidxs)}
#endf a2b_data_index


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



def transfer_milestone(adata, bdata, cumsum_name='cumsum_mean', knn_k=100, pca='X_pca', stream_tree='stream_tree', n_jobs=100):
    """
    adata is the subset stream tree object
    bdata is the full adata object
    calculate the adata.uns['milestone_df']
    """
    ## 1. calculate cumsum_mean to get cumsum coordinate of each cell
    node_cumsum_mean(adata, cumsum_name=cumsum_name)

    ## 2. calculate k-nearest neighbors using bdata pca
    print(datetime.now(), "knn...")
    nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='ball_tree', n_jobs=n_jobs).fit(bdata.obsm[pca])
    print(datetime.now(), "knn done.")
    indist, inind = nbrs.kneighbors(bdata.obsm[pca])

    ## 3. calculate index mapping for adata and bdata
    ddd = a2b_data_index(adata,bdata, list(range(adata.n_obs)))
    rddd = {v:k for k,v in ddd.items()}

    ## 4. calculate new cumsum dictionary
    cumsum_dict = {i: adata.obsm[cumsum_name][i, :] for i in range(adata.n_obs)}
    is_neigbor_set = set()
    ncumsum_dict = {}
    print(datetime.now(), "cumsum calc...")
    for i in range(bdata.n_obs):
        if i in rddd.keys(): ## already in subset
            ncumsum_dict[i] = cumsum_dict[rddd[i]]
            is_neigbor_set.add(i)
            continue

        nns = inind[i, 1:(knn_k+1)]
        u = set(nns) & set(rddd.keys())
        length = len(u)
        if length > 0:
            aidxs = [rddd[i] for i in u]
            is_neigbor_set.add(i)
            ncumsum_dict[i] = np.mean([cumsum_dict[i] for i in aidxs], axis=0)
        else:
            raise Exception('two few neigbors calculated, please increase')
    print(datetime.now(), "cumsum done.")

    bdata.obsm[cumsum_name] = np.array([ncumsum_dict[i] for i in range(len(ncumsum_dict))])
    bdata.uns[stream_tree] = adata.uns[stream_tree]

    ## 5. calculate percentage of each cells of bdata to the stream_tree branches
    df = stream_tree_order_end_nodes(bdata.uns[stream_tree])
    tree_milestone_belonging(bdata, stream_tree,df_branches=df)
    milestone_df = get_milestone_percentage(bdata, 'cumsum_percent')
    bdata.uns['milestone_df'] = milestone_df

    return bdata
#endf transfer_milestone
