import os
import re
import scipy
import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
from typing import Union, List, Tuple, Dict, Set
from .tree_utils import (_edge_two_ends,
                         _edgefreq_to_nodefreq,
                         tree_original_dict,
                         tree_unique_node,
                         remove_duplicated_index,
                         get_tree_leaves_attr,
                         get_markers_df,
                         TF_to_genes)
from ..util import get_quantiles



def tree_branch_against(adata, tree='fate_tree', branch="27-PODO"):
    """
    Given a branch end leave, search for all leaves against this branch the same level.

    1. get_tree_leaves_attr to get node_id to name
    2. find branching point
    3. find all leaves launching from the branching.
    4. get the against end leaves

    Parameters
    ----------
    adata: AnnData
        AnnData object
    tree: str
        key in adata.uns to get the fate tree
    branch: str
        the branch to search against

    Returns
    -------
    against_leaves: list
    """
    fate_tree = adata.uns[tree]
    dic = get_tree_leaves_attr(fate_tree)
    rdic = {v:k for k,v in dic.items()}
    branch_id = rdic.get(branch, None)
    if not branch_id:
        raise Exception(f"branch is not in the tree")

    branching = find_last_branching(fate_tree, branch_id)

    compare_leaves = find_branch_leaves(fate_tree, branching)

    compare_leaves = [dic[i] for i in compare_leaves]
    compare_leaves = [i for i in compare_leaves if i != branch]


    return compare_leaves



def find_branch_leaves(fate_tree, start):
    """
    give a start node, find all leaves

    Parameters
    ----------
    fate_tree: nx.DiGraph
        fate tree
    start: str
        the start node

    Returns
    -------
    leaves: list
    """

    group_dic = _divide_nodes_to_branches(fate_tree, start)
    lst = []
    for k,v in group_dic.items():
        if fate_tree.out_degree(v[-1]) >0:
            continue
        lst.append(v[-1])

    return lst


def find_samelevel_daugthers(fate_tree:nx.DiGraph=None, start_node:str=None):
    """
    give a node, find it's first branching node for each sub-branch

    Parameters
    ----------
    fate_tree: networkx.DiGraph
        fate tree
    start_node: str
        the node to find the branching point to get all daughter branches at same level
    """

    tranversed_nodes = list(nx.bfs_tree(fate_tree, start_node))#'0_0'
    start_prefix = start_node.split("_")[0]
    tranversed_nodes = [node for node in tranversed_nodes if not node.startswith(start_prefix)] #ignore main branch
    prefix_list = [node.split("_")[0] for node in tranversed_nodes]
    prefix_set = set()
    for i,p in enumerate(prefix_list):
        if p not in prefix_set:
            prefix_set.add(p)
        else:
            break
    return tranversed_nodes[:i]
#endf find_samelevel_daugthers

def time_slot_sets(node_list):
    """
    nodes format: idx_timeslot
    given a node_list, compile an asending list(by time), each element is a set of nodes in the same time slot

    Parameters
    ----------
    node_list: list
        list of nodes format: idx_timeslot
    """
    node_list = list(node_list)
    appendix = [int(node.split("_")[1]) for node in node_list]
    idxs = np.argsort(appendix)
    #print("idxs: ", idxs)
    curr_idx = idxs[0]
    ret_list = []
    for idx in idxs:
        if appendix[idx] != appendix[curr_idx]:
            ret_list.append(set([node_list[idx]]))
        elif len(ret_list) == 0:
            ret_list.append(set([node_list[idx]]))
        else:
            ret_list[-1].add(node_list[idx])
        curr_idx = idx
    return ret_list
#endf times slot

def add_merge_node(fate_tree, new_tree, father, nodes, name, f_original):
    """
    Add a new node with attributes attached.
    Currently only add attribute: ecount, original.

    Parameters
    ----------
    fate_tree: nx.DiGraph
        original fate_tree
    new_tree: nx.DiGraph
        new fate_tree by merge all subbranches.
    father: str
        the father node of the new node
    nodes: set
        nodes to be merged, mainly add up the edges counting
    name: str
        name of the new node
    f_original: str
        original name for the predix using in the new node.
    """
    from collections import Counter
    from functools import reduce
    new_tree.add_edge(father, name)
    #merge ecount
    #if fate_tree.has_node()
    d_list = [Counter(dict(fate_tree.nodes[n]['ecount'])) for n in nodes]
    d_ecount = reduce(lambda a,b: a+b, d_list)
    ecount = [(k,v) for k,v in d_ecount.items()]
    nx.set_node_attributes(new_tree, {name:ecount}, name='ecount')

    #set original
    original = (f_original[0], int(name.split('_')[1]))
    nx.set_node_attributes(new_tree, {name:original}, name='original')
    #TODO:
        #merge X_pca_ddhodge
        #merge u
        #merge cumsum
        #merge X_pac
    return new_tree
#endf add_merge_node


def helping_submerged_tree(adata, fate_tree='fate_tree', start_node='0_0', outname='fate_main_tree', iscopy=False):
    """
    from tree:
                             |---------------
          |------------------|
          |                  |------
          |              |------------
    ------|--------------|
          |              |------------
          |       |------------------
          |-------|
                  |-------------

    to tree:
          |------------------|---------------
          |
          |
    ------|--------------|------------
          |
          |
          |-------|------------------

    Parameters
    ----------
    adata: AnnData
        adata object where fate_tree in adata.uns
    fate_tree: str
        key of fate_tree in adata.uns
    start_node: str
        the node to find the branching point to get all daughter branches at same level
    outname: str
        key of the new fate_tree in adata.uns
    iscopy: bool
        if True, return a copy of adata, otherwise, inplace
    """

    adata = adata.copy() if iscopy else adata
    fate_tree = adata.uns[fate_tree]

    father = find_branch_end(fate_tree, start_node) #0_0
    daughter_nodes = find_samelevel_daugthers(fate_tree, father) #1_1, 3_1, 4_1
    merge_branch_dict = {}# key:val = last_node:its_merged nodes
    for d_node in daughter_nodes:
        bnodes = list(nx.dfs_tree(fate_tree, d_node)) #1_1, 1_2, ...
        bfather = bnodes[0] ##1_1

        bpoint = find_branch_start(fate_tree, bfather) ## 1_8
        bs_idx = np.where(np.array(bnodes) == bpoint)[0][0] ## 1_8 index
        ##1_8: [{5_9,8_9},{5_11},{5_14, 8_14}]
        merge_branch_dict[bnodes[bs_idx]] = time_slot_sets(bnodes[bs_idx+1:])

    ## 1.new_tree to remove nodes need to be merges
    new_tree = fate_tree.copy()
    for k, v in merge_branch_dict.items():
        rm_list = [j for i in v for j in i]
        for n in rm_list:
            new_tree.remove_node(n)
    ## 2. add new nodes with same prefix as father 1_8
    for k, v in merge_branch_dict.items():
        father = k
        prefix = k.split("_")[0]
        ## for a time slot like 9, merge 5_9 and 8_9 into 1_9
        for aset in v:
            appendix = list(aset)[0].split('_')[1]
            new_node_name = f'{prefix}_{appendix}'
            if fate_tree.has_node(father):
                add_merge_node(fate_tree, new_tree, father, aset, new_node_name, fate_tree.nodes[father]['original']) ## connect to last existing node
            else:
                add_merge_node(fate_tree, new_tree, father, aset, new_node_name, new_tree.nodes[father]['original'])  ## connect to the new node
            father = new_node_name
    adata.uns[outname] = new_tree
    return adata if iscopy else None
#endf helping_submerged_tree



def helping_merged_tree(adata, fate_tree='fate_tree', start_node='0_0', outname='fate_main_tree', iscopy=False):
    """
    from tree:
                             |---------------
          |------------------|
          |                  |------
          |              |------------
    ------|--------------|
          |              |------------
          |       |------------------
          |-------|
                  |-------------

    to tree:
          |------------------|---------------
          |
          |
    ------|--------------|------------
          |
          |
          |-------|------------------

    Parameters
    ----------
    adata: AnnData
        adata object where fate_tree in adata.uns
    fate_tree: str
        key of fate_tree in adata.uns
    start_node: str
        the node to find the branching point to get all daughter branches at same level
    outname: str
        key of the new fate_tree in adata.uns
    iscopy: bool
        if True, return a copy of adata, otherwise, inplace
    """

    adata = adata.copy() if iscopy else adata
    fate_tree = adata.uns[fate_tree]

    father = find_branch_end(fate_tree, start_node) #0_0
    daughter_nodes = find_samelevel_daugthers(fate_tree, father) #1_1, 3_1, 4_1
    merge_branch_dict = {}# key:val = last_node:its_merged nodes
    for d_node in daughter_nodes:
        bnodes = list(nx.dfs_tree(fate_tree, d_node)) #1_1, 1_2, ...
        bfather = bnodes[0] ##1_1

        bpoint = find_branch_end(fate_tree, bfather) ## 1_8
        bs_idx = np.where(np.array(bnodes) == bpoint)[0][0] ## 1_8 index
        ##1_8: [{5_9,8_9},{5_11},{5_14, 8_14}]
        merge_branch_dict[bnodes[bs_idx]] = time_slot_sets(bnodes[bs_idx+1:])

    ## 1.new_tree to remove nodes need to be merges
    new_tree = fate_tree.copy()
    for k, v in merge_branch_dict.items():
        rm_list = [j for i in v for j in i]
        for n in rm_list:
            new_tree.remove_node(n)
    ## 2. add new nodes with same prefix as father 1_8
    for k, v in merge_branch_dict.items():
        father = k
        prefix = k.split("_")[0]
        ## for a time slot like 9, merge 5_9 and 8_9 into 1_9
        for aset in v:
            appendix = list(aset)[0].split('_')[1]
            new_node_name = f'{prefix}_{appendix}'
            if fate_tree.has_node(father):
                add_merge_node(fate_tree, new_tree, father, aset, new_node_name, fate_tree.nodes[father]['original']) ## connect to last existing node
            else:
                add_merge_node(fate_tree, new_tree, father, aset, new_node_name, new_tree.nodes[father]['original'])  ## connect to the new node
            father = new_node_name
    adata.uns[outname] = new_tree
    return adata if iscopy else None
#endf helping_merged_tree



def find_a_branch_all_predecessors(fate_tree:nx.DiGraph=None,
                                   daughter_node:str=None):
    """
    find all predecessors of a node of same family, i.e. all nodes with the same prefix
    return list starts with the most grand father node

    Parameters
    ----------
    fate_tree: networkx.DiGraph
        fate tree
    daughter_node: str
        the node to find all predecessors
    """
    ret_list = [daughter_node]
    prefix = daughter_node.split("_")[0] + '_'
    current_daughter = daughter_node
    while True:
        father = list(fate_tree.predecessors(current_daughter))
        if len(father) != 1:
            break
        if father[0].startswith(prefix): ##only extract nodes from same family
            ret_list.append(father[0])
            current_daughter = father[0]
        else:
            break
    return ret_list[::-1]
#endf find_a_branch_all_predecessors

def find_branch_start(tree:nx.DiGraph=None,
                    current_node:str=None):
    """
    find the leaf from current

    Parameters
    ----------
    tree: networkx.DiGraph
        fate tree
    current_node: str
        the node to find the begining
    """
    node_list = list(nx.dfs_tree(tree, current_node))
    prefix = node_list[0].split("_")[0] + "_"
    node_list = [node for node in node_list if node.startswith(prefix)]
    return node_list[0]
#endf find_branch_start


def find_branch_end(tree:nx.DiGraph=None,
                    current_node:str=None):
    """
    find the leaf from current

    Parameters
    ----------
    tree: networkx.DiGraph
        fate tree
    current_node: str
        the node to find the leaf
    """
    node_list = list(nx.dfs_tree(tree, current_node))
    prefix = node_list[0].split("_")[0] + "_"
    node_list = [node for node in node_list if node.startswith(prefix)]
    return node_list[-1]
#endf find_branch_end




def find_last_branching(tree:nx.DiGraph=None,
                        daughter_node:str=None):
    """
    find all predecessors of a node of same family, i.e. all nodes with the same prefix
    return list starts with the most grand father node

    Parameters
    ----------
    tree: networkx.DiGraph
        fate tree
    daughter_node: str
        the node to find all predecessors
    """
    ret_list = [daughter_node]
    prefix = daughter_node.split("_")[0] + '_'
    current_daughter = daughter_node
    while True:
        father = list(tree.predecessors(current_daughter))
        if len(father) != 1:
            break
        if father[0].startswith(prefix): ##only extract nodes from same family
            ret_list.append(father[0])
            current_daughter = father[0]
        else:
            break
    father = list(tree.predecessors(current_daughter))
    return father[0]
#endf find_last_branching


def _divide_nodes_to_branches(tree:nx.DiGraph=None,
                              branching_node:str=None):
    """
        |--o--o--o---o--o--o--o--o---a
        |
        |
   -o---|ab
        |
        |--o-o--o--o-o---o-----o---o---o----b

    starts with ab nodes, divide the nodes to branches

    Parameters
    ----------
    tree: networkx.DiGraph
        fate tree
    branching_node: str
        the branching node to have branches, here is ab
    """
    from collections import defaultdict

    travesed_nodes = list(nx.dfs_tree(tree, branching_node, depth_limit=1000000))
    travesed_nodes = [node for node in travesed_nodes if node != branching_node]

    node_groups = defaultdict(list)
    for tn in travesed_nodes:
        node_groups[tn.split("_")[0]].append(tn)
    #assert(len(node_groups) == 2)
    return node_groups
#endf _divide_nodes_to_branches


def _edgefreq_to_nodefreq(edge_freq:List[tuple]=None,
                          d_edge2node:Dict=None):
    """
    from edge frequency to node frequency
    for each edge, it has two nodes, the node frequency is the sum of edge frequency

    Parameters
    ----------
    edge_freq: List[tuple]
        edge frequencies
    d_edge2node: Dict
        edge to node dictionary
    """
    from collections import defaultdict
    d_node_freq = defaultdict(int)
    for x, y in edge_freq:
        #print(x,y)
        a,b = d_edge2node.get(x, None)
        d_node_freq[a] += y
        d_node_freq[b] += y

    return d_node_freq
#endf _edgefreq_to_nodefreq


def tree_nodes_markers(adata: AnnData,
                       nodes1:Union[str, List[str]] = None,
                       nodes2:Union[str, List[str]] = None,
                       fate_tree = 'fate_tree',
                       vs1_name:str = None,
                       vs2_name:str = None,
                       vs_name:str = None,
                       iscopy = False,
                       **kwargs):
    """
    use scanpy to find markers
    node1 node2 name all from fate_tree, where ecount stored the edge count
    1. find the cells in node1 and node2 by counting edges two endings
    2. multiple the frequncy of each cell by the edge frequency
    3. construct each array for node1 and node2
    4. create a new AnnData(bdata) for using merged node1 and node2 expression matrix
    5. find markers for node1 and node2 in bdata
    6. shrink the bdata size with only 3 columns
    7. store bdata in adata.uns

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix with cell attributes.
    nodes1: str
        nodes1 names from fate_tree
    nodes2: str
        nodes2 names from fate_tree
    fate_tree: str
        fate tree name in adata.uns
    iscopy: bool
        if True, return a copy of adata, otherwise, update adata
    vs1_name: str
        the name of vs1, default is nodes1
    vs2_name: str
        the name of vs2, default is nodes2
    vs_name: str
        if specified, use this name as the name of vs1 and vs2
    kwargs:
        additional parameters for sc.tl.rank_genes_groups
    """
    adata = adata.copy() if iscopy else adata
    if nodes1 is None:
        raise ValueError('nodes1 is None')
    if nodes2 is None:
        raise ValueError(f"nodes2 is None")

    if isinstance(nodes1, str):
        nodes1 = [nodes1]
    if isinstance(nodes2, str):
        nodes2 = [nodes2]
    for node2 in nodes2:
        if node2 not in adata.uns[fate_tree].nodes():
            raise ValueError(f"{node2} is not in fate_tree")
    for node1 in nodes1:
        if node1 not in adata.uns[fate_tree].nodes():
            raise ValueError(f"{node1} is not in fate_tree")

    if not vs1_name:
        vs_name = f"{vs1_name}_vs_{vs2_name}" if vs1_name is not None and vs2_name is not None else None
    d_edge2node = _edge_two_ends(adata)
    ## construct nodes1 array
    nodes1_name = nodes1
    if not isinstance(nodes1, str):
        nodes1_name = '.'.join(nodes1)
        nodes1_tuple = []
        for node1 in nodes1:
            node1_tuple = adata.uns[fate_tree].nodes[node1]['ecount']
            nodes1_tuple.extend(node1_tuple)
    ## construct node1 array
    else:
        nodes1_tuple = adata.uns[fate_tree].nodes[nodes1]['ecount']

    d_nodes_nodes1 = _edgefreq_to_nodefreq(edge_freq=nodes1_tuple, d_edge2node=d_edge2node)
    X_nodes1 = adata.X[list(d_nodes_nodes1.keys()),:]
    obs_name_nodes1 = [f"{x}-n1" for x in  adata.obs_names[list(d_nodes_nodes1.keys())]]
    if scipy.sparse.issparse(X_nodes1):
        X_nodes1 = X_nodes1.toarray()
    X_nodes1 = X_nodes1 * np.array(list(d_nodes_nodes1.values()))[:, np.newaxis]



    ## construct node2 array
    nodes2_name = nodes2
    if not isinstance(nodes2, str):
        nodes2_name = '.'.join(nodes2)
        nodes2_tuple = []
        for node2 in nodes2:
            node2_tuple = adata.uns[fate_tree].nodes[node2]['ecount']
            nodes2_tuple.extend(node2_tuple)
    ## construct node2 array
    else:
        nodes2_tuple = adata.uns[fate_tree].nodes[nodes2]['ecount']

    d_nodes_nodes2 = _edgefreq_to_nodefreq(edge_freq=nodes2_tuple, d_edge2node=d_edge2node)
    X_nodes2 = adata.X[list(d_nodes_nodes2.keys()),:]
    obs_name_nodes2 = [f"{x}-n2" for x in  adata.obs_names[list(d_nodes_nodes2.keys())]]
    if scipy.sparse.issparse(X_nodes2):
        X_nodes2 = X_nodes2.toarray()
    X_nodes2 = X_nodes2 * np.array(list(d_nodes_nodes2.values()))[:, np.newaxis]

    vs1_name = nodes1_name if vs1_name is None else vs1_name
    vs2_name = nodes2_name if vs2_name is None else vs2_name

    # concat and initial a new anndata
    group = [vs1_name] * X_nodes1.shape[0]  + [vs2_name]*X_nodes2.shape[0]
    #print("shapes: ", X_nodes1.shape, X_nodes2.shape)
    X_merge = pd.DataFrame(np.concatenate((X_nodes1, X_nodes2)))
    X_merge.columns = adata.var_names
    X_merge.index = obs_name_nodes1 + obs_name_nodes2
    ## normalize columns to 1 avoid too large value
    normalized_X=(X_merge-X_merge.min())/(X_merge.max()-X_merge.min())


    bdata = sc.AnnData(normalized_X)
    bdata.obs['compare'] = group

    # find markers using scanpy
    sc.tl.rank_genes_groups(bdata, groupby='compare', groups=[vs1_name], reference=vs2_name, **kwargs)
    # shrink the helping anndata object
    bdata = bdata[:, :3]## make it smaller for storage.
    # save the result to adata
    if vs_name is None:
        vs_name = f"markers_{nodes1_name}_vs_{nodes2_name}"
    else:
        vs_name = f"markers_{vs_name}"
    print("vs_name: ", vs_name)
    adata.uns[vs_name] = bdata
    return adata if iscopy else None
#endf tree_nodes_markers



def tree_branches_markers(adata: AnnData,
                          branching_node:str,
                          branch_1: str,
                          branches_2: List[str] = None,
                          include_pre_branch: bool = False,
                          name_append: str = None,
                          ratio:float=0.3,
                          fate_tree = 'fate_tree',
                          iscopy:bool=False,
                          **kwargs):

    """
        |<<O<<<<<------O-----a
        |
    --<<|ab
        |             |----------O----b
        |>>O>>>>------|bc
        |             |------------O----c
        |
        |<<O<<<<<---------------O----d


    Take differentiation of branch bc against branch a, d and branching ab
    first find all branches(a,d,bc)
    perform differential analysis between bc against (a, d and ab) if include_pre_branch is True
    perform differential analysis between bc against (a, d) if include_pre_branch is False
    (>>>) agains (<<<)

    Parameters
    ----------
    adata: AnnData
        AnnData object
    branching_node: str
        the branching node
    branches_1: str
        the branch to be compared
    branches_2: List[str]
        the branches to be compared with, if None, all other branches will be used
    include_pre_branch: bool
        whether to include the branching node branch in the comparison
    ratio: float
        the ratio of the number of cells in the branch to be compared to the number of cells in the other branches
    fate_tree: str
        the fate tree name
    iscopy: bool
        whether to return a copy of the AnnData object
    kwargs:
        additional arguments passed to tree_nodes_markers

    """
    import functools

    node_groups = _divide_nodes_to_branches(adata.uns[fate_tree], branching_node)
    #print(node_groups)
    ## branch_1 is defined in here
    branch_1_predix = branch_1.split('_')[0]
    assert branch_1_predix in node_groups.keys(), f"branch_1 {branch_1} is not in the tree"


    ## branches_2 can be defined by user or automatically
    if branches_2 is None:
        branches_2_predix = [k for k in node_groups.keys() if k != branch_1_predix]
    else:
        branches_2_predix = [k for k in branches_2 if k in node_groups.keys()]

    ## get parts of nodes for each branch
    for k in node_groups:
        len_branch = max(int(len(node_groups[k])*ratio), 1)
        node_groups[k] = node_groups[k][:len_branch]

    ## add branching node to branches_2_predix
    if include_pre_branch:
        pre_branch = find_a_branch_all_predecessors(adata.uns[fate_tree], branching_node)
        len_pre_branch = max(int(len(pre_branch)*ratio), 1)
        node_groups["branching"] = (pre_branch[-1*len_pre_branch:])
        branches_2_predix.append("branching")

    adata = adata.copy() if iscopy else adata

    nodes1 = node_groups[branch_1_predix]
    nodes2 = functools.reduce(lambda x,y: x+y, [node_groups[k] for k in branches_2_predix])


    vs2_name = f"{branching_node}_rest" if include_pre_branch else f"rest"
    if name_append is not None:
        vs2_name = f"{vs2_name}_{name_append}"

    tree_nodes_markers(adata, nodes1, nodes2,  fate_tree=fate_tree, vs1_name=f"{branch_1}", vs2_name=vs2_name,**kwargs)
    return adata if iscopy else None
#endf tree_branches_markers


#def tree_full_mbranch_markers():
#    """
#    merge each branch for main branches comparison
#    Idea:
#        1. Merge fate_tree by the time slot number
#        2. We next can use top percent of cells to do the comparison
#    """
#    pass
#
##endf tree_full_mbranch_markers




def tree_mbranch_markers(adata: AnnData,
                         branches_1: set([tuple]),
                         branches_2: set([tuple]),
                         tree_attr:str="original",
                         fate_tree:str='fate_tree',
                         include_pre_branch: bool = False,
                         name_append: str = "",
                         ratio:float=0.3,
                         compare_position:str = "start", # or end
                         vs_name:str=None,
                         iscopy:bool=False,
                         **kwargs):

    """
    if compare_position is start:

            |>>O>>>>>>>----O-----a
            |
        -<<<|ab
            |
            |<<O<<<<<<<<------------O----b
            |
            |>>O>>>>>>>>------------O----c

    if compare_position is end:

            |--O-----------O<<<<<a
            |
            |
        --<<|ab
            |
            |--O------------->>>>>>>O>>>>b
            |
            |--O------------->>>>>>>O>>>>c

    1. find different markers that regulate branch set(a) and branch set(b, c)
    2. the markers can be the start of the branch or the end of the branch
    3. this function focus on the start of the branch
        i)   find all of the nodes of the branch a and branch b
        ii)  select a good ratio of the nodes as the start of the branch(>>>>)
        iii) find the markers that regulate the start of the branch

    (>>>>>) against (<<<)

    Parameters
    ----------
    adata: AnnData
        AnnData object
    branch_1: set
        the branching node to have branches, here could be a
    branch_2: set
        the branching node to have branches, here could be b
    tree_attr: str
        the attribute of the tree to be used: default is node_name, could also be original
    fate_tree: str
        the fate tree name
    include_pre_branch: bool
        whether to include the branching node branch in the comparison
    ratio: float
        the ratio of the nodes in a branch to be compared to the number of nodes in the other branch
    compare_position: str
        the position to compare, could be `start` or `end`
    iscopy: bool
        whether to return a copy of the AnnData object
    kwargs: dict
        the parameters for tree_nodes_markers
    """

    if not isinstance(branches_1, set):
        raise ValueError("branches_1 must be a set")
    if not isinstance(branches_2, set):
        raise ValueError("branches_2 must be a set")


    nodes1_list = []
    nodes2_list = []

    for branch_1 in branches_1:
        if tree_attr == "node_name":
            pass
        elif tree_attr == "original":
            #convert original to node_name
            branch_1_d = tree_original_dict(adata.uns[fate_tree], branch_1)
            if len(branch_1_d) <1 :
                raise Exception(f"branch_1 is not in the tree")
            branch_1 = find_branch_end(adata.uns[fate_tree], list(branch_1_d.keys())[0])
        nodes1 = find_a_branch_all_predecessors(adata.uns[fate_tree], branch_1)
        nodes1_list.extend(nodes1)

    for branch_2 in branches_2:
        if tree_attr == "node_name":
            pass
        elif tree_attr == "original":
            #convert original to node_name
            branch_2_d = tree_original_dict(adata.uns[fate_tree], branch_2)
            if len(branch_2_d) <1:
                raise Exception(f"branch_1 or branch_2 is not in the tree")
            branch_2 = find_branch_end(adata.uns[fate_tree], list(branch_2_d.keys())[0])
        nodes2 = find_a_branch_all_predecessors(adata.uns[fate_tree], branch_2)
        nodes2_list.extend(nodes2)

    nodes1 = nodes1_list
    nodes2 = nodes2_list

#    branching_node1 = find_last_branching(adata.uns[fate_tree], branch_1)
#    branching_node2 = find_last_branching(adata.uns[fate_tree], branch_2)
#
#
#    assert branching_node1 == branching_node2, "branching node is not the same"

    len1 = max(int(len(nodes1)*ratio),1)
    len2 = max(int(len(nodes2)*ratio),1)
    if compare_position == "start":
        nodes1 = nodes1[:len1]
        nodes2 = nodes2[:len2]
    elif compare_position == "end":
        nodes1 = nodes1[-1*len1:]
        nodes2 = nodes2[-1*len2:]
    else:
        raise Exception(f"compare_position {compare_position} is not supported, options: start, end")

#    if include_pre_branch:
#        pre_branch = find_a_branch_all_predecessors(adata.uns[fate_tree], branching_node1)
#        len_pre_branch = max(int(len(pre_branch)*ratio), 1)
#        nodes_branching = (pre_branch[-1*len_pre_branch:])
#        nodes2 += nodes_branching


    adata = adata.copy() if iscopy else adata
    against_name = f"{branch_2}" if not name_append else f"{branch_2}_{name_append}"
    tree_nodes_markers(adata, nodes1, nodes2, fate_tree=fate_tree,  vs1_name=f"{branch_1}", vs2_name=against_name, vs_name=vs_name, **kwargs)
    return adata if iscopy else None
#endf tree_mbranch_markers


def tree_2branch_markers(adata: AnnData,
                         branch_1: Union[str, tuple] ,
                         branch_2: Union[str, tuple],
                         tree_attr:str="original",
                         fate_tree:str='fate_tree',
                         include_pre_branch: bool = False,
                         name_append: str = "",
                         ratio:float=0.3,
                         compare_position:str = "start", # or end
                         vs_name:str=None,
                         iscopy:bool=False,
                         **kwargs):

    """
    if compare_position is start:

            |>>O>>>>>>>----O-----a
            |
        -<<<|ab
            |
            |<<O<<<<<<<<------------O----b

    if compare_position is end:

            |--O-----------O<<<<<a
            |
            |
        --<<|ab
            |
            |--O------------->>>>>>>O>>>>b

    1. find different markers that regulate branch a and branch b
    2. the markers can be the start of the branch or the end of the branch
    3. this function focus on the start of the branch
        i)   find all of the nodes of the branch a and branch b
        ii)  select a good ratio of the nodes as the start of the branch(>>>>)
        iii) find the markers that regulate the start of the branch

    (>>>>>) against (<<<)

    Parameters
    ----------
    adata: AnnData
        AnnData object
    branch_1: str
        the branching node to have branches, here could be a
    branch_2: str
        the branching node to have branches, here could be b
    tree_attr: str
        the attribute of the tree to be used: default is node_name, could also be original
    fate_tree: str
        the fate tree name
    include_pre_branch: bool
        whether to include the branching node branch in the comparison
    ratio: float
        the ratio of the nodes in a branch to be compared to the number of nodes in the other branch
    compare_position: str
        the position to compare, could be `start` or `end`
    iscopy: bool
        whether to return a copy of the AnnData object
    kwargs: dict
        the parameters for tree_nodes_markers
    """
    if tree_attr == "node_name":
        pass
    elif tree_attr == "original":
        #convert original to node_name
        branch_1_d = tree_original_dict(adata.uns[fate_tree], branch_1)
        branch_2_d = tree_original_dict(adata.uns[fate_tree], branch_2)
        #print(branch_1_d)
        #print(branch_2_d)
        if len(branch_1_d) <1 or len(branch_2_d) <1:
            raise Exception(f"branch_1 or branch_2 is not in the tree")

        branch_1 = find_branch_end(adata.uns[fate_tree], list(branch_1_d.keys())[0])
        branch_2 = find_branch_end(adata.uns[fate_tree], list(branch_2_d.keys())[0])


    #print(branch_1, branch_2)
    nodes1 = find_a_branch_all_predecessors(adata.uns[fate_tree], branch_1)
    nodes2 = find_a_branch_all_predecessors(adata.uns[fate_tree], branch_2)
    print("nodes1: ", nodes1)
    print("nodes2: ", nodes2)

    branching_node1 = find_last_branching(adata.uns[fate_tree], branch_1)
    branching_node2 = find_last_branching(adata.uns[fate_tree], branch_2)


    assert branching_node1 == branching_node2, "branching node is not the same"

    len1 = max(int(len(nodes1)*ratio),1)
    len2 = max(int(len(nodes2)*ratio),1)
    if compare_position == "start":
        nodes1 = nodes1[:len1]
        nodes2 = nodes2[:len2]
    elif compare_position == "end":
        nodes1 = nodes1[-1*len1:]
        nodes2 = nodes2[-1*len2:]
    else:
        raise Exception(f"compare_position {compare_position} is not supported, options: start, end")

    if include_pre_branch:
        pre_branch = find_a_branch_all_predecessors(adata.uns[fate_tree], branching_node1)
        len_pre_branch = max(int(len(pre_branch)*ratio), 1)
        nodes_branching = (pre_branch[-1*len_pre_branch:])
        nodes2 += nodes_branching


    adata = adata.copy() if iscopy else adata
    against_name = f"{branch_2}" if not name_append else f"{branch_2}_{name_append}"
    tree_nodes_markers(adata, nodes1, nodes2, fate_tree=fate_tree, vs1_name=f"{branch_1}", vs2_name=against_name, vs_name=vs_name, **kwargs)
    return adata if iscopy else None
#endf tree_2branch_markers


def tree_markers_dump_table(adata: AnnData,
                            name:str='markers_3_97_vs_0_11_rest_0.3withParents',
                            filename:str="x.csv"):
    """
    dump markers to tables:
    supported format: csv, xls, xlsx, tsv, txt

    Parameters
    ----------
    adata: AnnData
        AnnData object
    name: str
        the name of the markers
    filename: str
        the filename of the table
    """

    if name not in adata.uns.keys():
        raise ValueError(f"{name} is not in adata.uns.keys()")

    rank_groups = adata.uns[name].uns['rank_genes_groups']
    df = pd.DataFrame({"names" : [i[0] for i in rank_groups['names'].tolist()],
                        "scores" : [i[0] for i in rank_groups['scores'].tolist()],
                        "pvals"  : [i[0] for i in rank_groups['pvals'].tolist()],
                        "pvals_adj" : [i[0] for i in rank_groups['pvals_adj'].tolist()],
                        "logfoldchanges" : [i[0] for i in rank_groups['logfoldchanges'].tolist()], }
                      )

    if filename.endswith("csv"):
        df.to_csv(filename)
    elif filename.endswith("xlsx") or filename.endswith("xls"):
        df.to_excel(filename)
    elif filename.endswith("tsv") or filename.endswith("txt"):
        df.to_csv(filename, sep="\t")
    else:
        raise ValueError(f"filename {filename} is not supported, support csv, xls, xlsx, tsv, txt")
#endf tree_markers_dump_table

def TF_gene_correlation(adata: AnnData,
                        tfadata: AnnData,
                        name:str='markers_3_97_vs_0_11_rest_0.3withParents'):
    """
    calculate the correlation between TF and genes for a comparison of markers
    This is only calculate the correlation without caring about the order of cells


    Parameters
    ----------
    adata: AnnData
        AnnData object with gene expression
    tfadata: AnnData
        AnnData object with TF binding data
    name: str
        the name of the differentiation branches
    """
    from collections import Counter, defaultdict
    from scipy.stats import pearsonr

    zdata = tfadata.uns[name].copy()
    group_obs =  zdata.obs_names
    group_obs1 = set([x[:-3] for x in zdata.obs_names if x.endswith('n1')])
    group_obs2 = set([x[:-3] for x in zdata.obs_names if x.endswith('n2')])
    compare_obs = list(group_obs1 | group_obs2)

    TFs = [x[0] for x in zdata.uns['rank_genes_groups']['names']]
    d_tf2gene = TF_to_genes(TFs, False)

    d_gene2tf= {}
    #for k, v in d_tf2gene.items():
    #    if k not in d_gene2tf:
    #        d_gene2tf[v] = k

    for k, v in d_tf2gene.items():
        #if k not in d_gene2tf:
        d_gene2tf[v] = d_gene2tf.get(v, []) + [k]
#    for k, v in d_gene2tf.items():
#        if len(v) > 1:
#            print(k, v)
    for k, v in d_gene2tf.items():
        if len(v) == 1:
            d_gene2tf[k] = v[0]
        elif len(v) > 1:
            if k in v:
                d_gene2tf[k] = k
            else:
                d_gene2tf[k] = v[0]
                if "VAR." in v[0]:
                    #print(k, v)
                    vv = re.sub(r"\(VAR\.\d+\)", "", v[0])
                    d_gene2tf[k] = vv

            #print(k, v)

    #d_gene2tf = {v: k for k, v in d_tf2gene.items()}
    #shared = [(i, d_tf2gene[i]) for i in adata.var_names if i in d_tf2gene]

    shared = [(i, d_gene2tf[i]) for i in adata.var_names if i in d_gene2tf]
    print(len(shared))
    shared_genes = np.array([i[0] for i in shared])
    shared_tfs = np.array([i[1] for i in shared])
    remove_duplicated_index = remove_duplicated_index(shared_genes)
    shared_genes = shared_genes[remove_duplicated_index]
    shared_tfs = shared_tfs[remove_duplicated_index]
    print(len(shared_genes))

    expression_mtx = adata[compare_obs, shared_genes].X
    TF_mtx = tfadata[compare_obs, shared_tfs].X

    d_corr = defaultdict(float)
    for idx, sym in enumerate(shared_tfs):
        expression = expression_mtx[:, idx].toarray().ravel()
        TF =  TF_mtx[:, idx].toarray().ravel()
        corr = pearsonr(expression, TF)[0]
        if np.isnan(corr):
            continue
        d_corr[sym] = pearsonr(expression, TF)[0]
    corr_ordered = sorted(d_corr.items(), key=lambda x:x[1], reverse=True)

    return corr_ordered
#endf TF_gene_correlation


def branch_TF_gene_correlation(tf_df:pd.DataFrame,
                               gene_df:pd.DataFrame):
    """
    calculate the correlation between TF and genes using pseudo time matrix
    return list of tuples recording the correlations ordered by correlation descendingly.

    Parameters
    ----------
    tf_df: pd.DataFrame
        the TF expression matrix, index is TF name, columns are bins
    gene_df: pd.DataFrame
        the gene expression matrix, index is gene name, columns are bins
    """
    from collections import Counter, defaultdict
    from scipy.stats import pearsonr


    TFs = tf_df.index.tolist()
    d_tf2gene = TF_to_genes(TFs, False)

    shared = list((k, v) for k, v in d_tf2gene.items() if v in gene_df.index)
    shared_tfs = np.array([i[0] for i in shared])
    shared_genes = np.array([i[1] for i in shared])

    tf_df = tf_df.loc[shared_tfs]
    gene_df = gene_df.loc[pd.unique(shared_genes)]

    d_corr = defaultdict(float)
    for idx, sym in enumerate(shared_tfs):
        TF = tf_df.loc[sym, :].values
        gene = d_tf2gene[sym]
        expression = gene_df.loc[gene, :].values
        corr = pearsonr(expression, TF)[0]
        if np.isnan(corr):
            continue
        d_corr[sym] = pearsonr(expression, TF)[0]
    corr_ordered = sorted(d_corr.items(), key=lambda x:x[1], reverse=True)
    #corr_ordered = [(k, v, d_tf2gene[k]) for k, v in corr_ordered]
    return corr_ordered
#endf branch_TF_gene_correlation




def branch_TF_gene_correlation_v0(tf_df:pd.DataFrame,
                               gene_df:pd.DataFrame):
    """
    calculate the correlation between TF and genes using pseudo time matrix
    return list of tuples recording the correlations ordered by correlation descendingly.

    Parameters
    ----------
    tf_df: pd.DataFrame
        the TF expression matrix, index is TF name, columns are bins
    gene_df: pd.DataFrame
        the gene expression matrix, index is gene name, columns are bins
    """
    from collections import Counter, defaultdict
    from scipy.stats import pearsonr
    ## All to upper
    tf_df.index = [x.upper() for x in tf_df.index]
    gene_df.index = [x.upper() for x in gene_df.index]
    ## shared symbols
    shared_symbol = list(set(tf_df.index) & set(gene_df.index))

    d_corr = defaultdict(float)
    for idx, sym in enumerate(shared_symbol):
        TF = tf_df.loc[sym, :].values
        expression = gene_df.loc[sym, :].values
        corr = pearsonr(expression, TF)[0]
        if np.isnan(corr):
            continue
        d_corr[sym] = pearsonr(expression, TF)[0]
    corr_ordered = sorted(d_corr.items(), key=lambda x:x[1], reverse=True)
    return corr_ordered
#endf branch_TF_gene_correlation_v0

def tree_branches_smooth_window(adata: AnnData,
                                start_branching_node: str="",
                                end_branching_node: str="",
                                tree_attr:str="original",
                                fate_tree: str= 'fate_tree',
                                smooth_window_ratio: float=0.1,
                                ):

    """
         |--O-----------O-----a
         |
         |
     >>>>|ab
         |
         |>>O>>>>>>>>>>>>>>>>>>>>O>>>>b

    start_branching_node is ab
    end_branching_node is b
    the function would traverse all nodes(bins) from ab to b (path:>>>>)
    average the expression of the each nodes by cells presented in this node.
    next smooth the by a smooth_window_ratio * len(number of bins)

    Parameters
    ----------
    adata: AnnData
        AnnData object with gene expression or TF binding data
    start_branching_node: str
        the name of the start branching node, if "", use the lastest branching node
    end_branching_node: str
        the name of the end branching node
    fate_tree: str
        the name of the fate tree
    smooth_window_ratio: float
        the ratio of the smooth window
    """

    if tree_attr == "node_name":
        pass
    elif tree_attr == "original":
        #convert original to node_name
        end_branching_node_d = tree_original_dict(adata.uns[fate_tree], end_branching_node)
        end_branching_node = find_branch_end(adata.uns[fate_tree], list(end_branching_node_d.keys())[0])
    if not start_branching_node:
        start_branching_node = find_last_branching(adata.uns[fate_tree], end_branching_node)

    if start_branching_node not in adata.uns[fate_tree].nodes:
        raise ValueError(f"start_branching_node {start_branching_node} is not in the fate tree")
    if end_branching_node not in adata.uns[fate_tree].nodes:
        raise ValueError(f"end_branching_node {end_branching_node} is not in the fate tree")

    bins = find_a_branch_all_predecessors(adata.uns[fate_tree], start_branching_node) + \
                    find_a_branch_all_predecessors(adata.uns[fate_tree], end_branching_node)

    #for now only use normlized data, ArchR uses counts

    d_edge2node = _edge_two_ends(adata)
    list_bin_expression = []
    for idx, bin_ in enumerate(bins):
        e_nodes = adata.uns[fate_tree].nodes[bin_]['ecount']
        d_nodes_nodes1 = _edgefreq_to_nodefreq(edge_freq=e_nodes, d_edge2node=d_edge2node)
        X_nodes1 = adata.X[list(d_nodes_nodes1.keys()),:].toarray()
        X_nodes1 = X_nodes1 * np.array(list(d_nodes_nodes1.values()))[:, np.newaxis]
        ## can change strategy in here. now is mean
        list_bin_expression.append(X_nodes1.mean(axis=0))
    mat = np.stack(list_bin_expression, axis=1) ## genes x bins

    ## smoothing: using window = 0.1 * total number of bins
    smooth_window_ratio = smooth_window_ratio
    window = int(np.ceil(len(bins)*smooth_window_ratio))
    smooth_mat = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window)/window, mode='same'), axis=1, arr=mat)
    smooth_df = pd.DataFrame(smooth_mat, index=adata.var_names, columns=bins)

    return smooth_df
#endf tree_branches_smooth_window


def tree_branches_expression_bins(adata: AnnData,
                                 start_branching_node: str="",
                                 end_branching_node: str="",
                                 tree_attr:str="original",
                                 fate_tree: str= 'fate_tree',
                                 lognorm = True,
                                 scale = False,
                                 ):

    """
         |--O-----------O-----a
         |
         |
     >>>>|ab
         |
         |>>O>>>>>>>>>>>>>>>>>>>>O>>>>b

    start_branching_node is ab
    end_branching_node is b
    the function would traverse all nodes(bins) from ab to b (path:>>>>)
    average the expression of the each nodes by cells presented in this node.

    Parameters
    ----------
    adata: AnnData
        AnnData object with gene expression or TF binding data
    start_branching_node: str
        the name of the start branching node, if "", use the lastest branching node
    end_branching_node: str
        the name of the end branching node
    fate_tree: str
        the name of the fate tree
    """

    if tree_attr == "node_name":
        pass
    elif tree_attr == "original":
        #convert original to node_name
        end_branching_node_d = tree_original_dict(adata.uns[fate_tree], end_branching_node)
        end_branching_node = find_branch_end(adata.uns[fate_tree], list(end_branching_node_d.keys())[0])
    if not start_branching_node:
        start_branching_node = find_last_branching(adata.uns[fate_tree], end_branching_node)

    if start_branching_node not in adata.uns[fate_tree].nodes:
        raise ValueError(f"start_branching_node {start_branching_node} is not in the fate tree")
    if end_branching_node not in adata.uns[fate_tree].nodes:
        raise ValueError(f"end_branching_node {end_branching_node} is not in the fate tree")

    bins = find_a_branch_all_predecessors(adata.uns[fate_tree], start_branching_node) + \
                    find_a_branch_all_predecessors(adata.uns[fate_tree], end_branching_node)

    #for now only use normlized data, ArchR uses counts

    d_edge2node = _edge_two_ends(adata)
    list_bin_expression = []
    for idx, bin_ in enumerate(bins):
        e_nodes = adata.uns[fate_tree].nodes[bin_]['ecount']
        d_nodes_nodes1 = _edgefreq_to_nodefreq(edge_freq=e_nodes, d_edge2node=d_edge2node)
        X_nodes1 = adata.X[list(d_nodes_nodes1.keys()),:].toarray()
        X_nodes1 = X_nodes1 * np.array(list(d_nodes_nodes1.values()))[:, np.newaxis]
        ## can change strategy in here. now is mean
        list_bin_expression.append(X_nodes1.mean(axis=0))
    mat = np.stack(list_bin_expression, axis=1) ## genes x bins
    mat = pd.DataFrame(mat, index=adata.var_names, columns=bins)
    if lognorm:
        ann = AnnData(mat.T)
        sc.pp.normalize_total(ann, target_sum=1e4)
        sc.pp.log1p(ann)
        if scale:
            sc.pp.scale(ann, max_value=10)
        mat = pd.DataFrame(ann.X.T, index=adata.var_names, columns=bins)
    return mat
#endf tree_branches_expression_bins


def tree_branches_cells(adata: AnnData,
                        start_branching_node: str="",
                        end_branching_node: str="",
                        tree_attr:str="original",
                        fate_tree: str= 'fate_tree',
                        bin_max_ratio = 0.01  ### each bin can contain no more than 1/100 of total cells
                        ):
    """
    goal: collection cells from a branch to, can with real time or pseudo time
    1. since a cell can show up multiple times, need to dedup.
    2. some cells show up too few times can use a cutoff to remove low frequency cells
    3. for plot LOWESS model to get smooth with stdev
    FOR NOW just implement only for real time, for this, the task is to select cells

    Parameters
    ----------
    adata: AnnData
        AnnData object with gene expression or TF binding data
    start_branching_node: str
        the name of the start branching node, if "", use the lastest branching node
    end_branching_node: str
        the name of the end branching node
    fate_tree: str
        the name of the fate tree
    bin_max_ratio: float
        each bin can contain no more than ratio of total cells

    Returns
    -------
    s: set
        a set of cells
    """
    if tree_attr == "node_name":
        pass
    elif tree_attr == "original":
        #convert original to node_name
        end_branching_node_d = tree_original_dict(adata.uns[fate_tree], end_branching_node)
        end_branching_node = find_branch_end(adata.uns[fate_tree], list(end_branching_node_d.keys())[0])
    if not start_branching_node:
        start_branching_node = find_last_branching(adata.uns[fate_tree], end_branching_node)

    if start_branching_node not in adata.uns[fate_tree].nodes:
        raise ValueError(f"start_branching_node {start_branching_node} is not in the fate tree")
    if end_branching_node not in adata.uns[fate_tree].nodes:
        raise ValueError(f"end_branching_node {end_branching_node} is not in the fate tree")

    bins = find_a_branch_all_predecessors(adata.uns[fate_tree], start_branching_node) + \
                    find_a_branch_all_predecessors(adata.uns[fate_tree], end_branching_node)

    #for now only use normlized data, ArchR uses counts

    d_edge2node = _edge_two_ends(adata)
    list_bin_expression = []
    s =  set() # contain cells need to check
    for idx, bin_ in enumerate(bins[::-1]):
        e_nodes = adata.uns[fate_tree].nodes[bin_]['ecount']
        d_nodes_nodes1 = _edgefreq_to_nodefreq(edge_freq=e_nodes, d_edge2node=d_edge2node)
        quant = 1- min(1,   (bin_max_ratio * adata.n_obs)/len(d_nodes_nodes1)) ### each bin can contain no more than 1/100 of total cells
        if quant == 0 and len(d_nodes_nodes1) > 10: ## keep all if no more than 10 cells
            quant = 0.2
        minv = np.quantile(list(d_nodes_nodes1.values()), q=quant)
        d_nodes_nodes1 = {k:v for k,v in d_nodes_nodes1.items() if v >= minv and k not in s}
        s = s | d_nodes_nodes1.keys() ## this would remove from large bins to small bins,
        #list_bin_expression.append(d_nodes_nodes1)## this would remove from large bins to small bins,

    return s
#endf tree_branches_cells



def branch_heatmap_matrix(tf_df:pd.DataFrame,
                          max_features:int=100,
                          var_cutoff:float=0.9,
                          label_markers:List=None):
    """
    order, filtering and scaling by rows of a pseudo time matrix
    select maximum max_features genes by variance
    if label_markers is offered, then the genes in label_markers would be selected first
    scale by rows and cutoff the valuse by abs(2)

    Parameters
    ----------
    tf_df: pd.DataFrame
        pseudo time matrix
    max_features: int
        maximum number of genes to be selected
    var_cutoff: float
        variance cutoff to select genes
    label_markers: List
        list of genes to be selected first
    """
    if label_markers is None:
        ## calculate var of each row
        rowvars = np.array(tf_df).var(axis=1)
        #varQ to do the filtering
        varq = get_quantiles(rowvars)
        varq_idx = np.argsort(rowvars)[::-1]
        tf_df = tf_df.iloc[varq_idx, :]

        n = len(tf_df)
        if max_features is not None:
            n = max_features
        elif var_cutoff is not None:
            n = len(tf_df) * (1-var_cutoff)
        tf_df = tf_df.iloc[:n, :]
    else:
        tf_df = tf_df.loc[label_markers, :]

    vectors = tf_df.values
    scaler = StandardScaler()
    scaled_rows = scaler.fit_transform(vectors.T).T
    scaled_rows[scaled_rows <-2] = -2
    scaled_rows[scaled_rows >2] = 2
    row_order = np.argmax(scaled_rows, axis=1)
    row_index = np.argsort(row_order)

    if label_markers is None:
        df = pd.DataFrame( scaled_rows[row_index, :], index=tf_df.index[row_index], columns=tf_df.columns)
        return df
    else:
        df = pd.DataFrame(scaled_rows, index=label_markers, columns=tf_df.columns)
        return df
#endf branch_heatmap_matrix



def branch_regulator_detect(adata:AnnData,
                            tfadata:AnnData,
                            branch,
                            tree = "fate_tree",
                            tree_attr="original",
                            ratio=0.5,
                            intersect_regulator=60,
                            vs_name="auto",
                            log2fc="auto",
                            correlation="auto",
        ):
    """
    Encapsulate the step of regulator finding
    1. specify an end branch name, automatically decide the comparison branch/branches
    2. perform differentation between two branches
    3. select top markers

    Parameters
    ----------
    adata: AnnData
        the data object
    tfadata: AnnData
        the data object of transcription factors
    branch: str
        the end branch name
    tree: str
        the fate tree name
    tree_attr: str
        the fate tree attribute
    ratio: float
        the ratio of cells in the branch to be selected
    intersect_regulator: int
        the number of intersected regulators from differentiation or correlations
    vs_name: str
        the name of the comparison branch default: {tree}_level_{branch}
    log2fc: Union[str, float]
        the log2fc cutoff for differentiation default: "auto"
    correlation: Union[str, float]
        the correlation cutoff for correlation default: "auto"
    """
    #decide the branch_2 name

    # if branch2 is a simple branch, just compare
    # else if branch 2 is branches of branches, need to merge to a single branch.

    branch_2 = tree_branch_against(tfadata, tree=tree, branch=branch)
    print("branch_2",branch_2)

    if len(branch_2) > 1: ## need merge tree

        ## get id of the  branching point
        dic = get_tree_leaves_attr(tfadata.uns[tree])
        rdic = {v:k for k,v in dic.items()}
        branch_id = rdic.get(branch, None)
        if not branch_id:
            raise Exception(f"branch is not in the tree")

        branching = find_last_branching(adata.uns[tree], branch_id)

        helping_submerged_tree(adata, fate_tree=tree, start_node=branching, outname=f'{tree}_level_{branch}', iscopy=False)
        helping_submerged_tree(tfadata, fate_tree=tree, start_node=branching, outname=f'{tree}_level_{branch}', iscopy=False)
        tree = f'{tree}_level_{branch}'



    #tree_2branch_markers(tfadata, branch_1=branch, branch_2=tuple(branch_2), tree_attr=tree_attr, ratio=ratio, vs_name=f'{branch}vs', fate_tree=tree, compare_position="end")
    tree_mbranch_markers(tfadata, branches_1=set([branch]), branches_2=set([tuple(branch_2)]), tree_attr=tree_attr, ratio=ratio, vs_name=f'{branch}vs', fate_tree=tree,compare_position="end")
    df = get_markers_df(tfadata, f'markers_{branch}vs').sort_values("logfoldchanges", ascending=False)

    if log2fc=="auto" or log2fc=="AUTO":
        df_top_names = df.names[:intersect_regulator] ## select 2 times of the differentiation
    else:
        df_top_names = df.names[(df.logfoldchanges > log2fc) & (df.pvals < 0.05)]

    ## calculate the correlation
    b_tf_traj_mat = tree_branches_smooth_window(tfadata, end_branching_node=branch, tree_attr=tree_attr)
    b_gene_traj_mat = tree_branches_smooth_window(adata, end_branching_node=branch, tree_attr=tree_attr)
    b_correlation = branch_TF_gene_correlation(b_tf_traj_mat, b_gene_traj_mat)
    b_correlation_df = pd.DataFrame.from_records(b_correlation, columns =['sym', 'score'], index=[i[0] for i in b_correlation])
    b_correlation_df.sort_values("score", ascending=False, inplace=True)

    #d_tf2gene = TF_to_genes(list(b_tf_traj_mat.index), ones=False)

    if correlation=="auto" or correlation=="AUTO":
        b_top_correlation = b_correlation_df.index[:intersect_regulator] ## select 2 times of the correlation
    else:
        b_top_correlation = [sym for sym, corr in b_correlation if corr > correlation]

    b_inter_TFs   = [i.upper() for i in list(set(b_top_correlation) & set(df_top_names))]
    b_correlation_df = b_correlation_df.loc[b_inter_TFs,:].sort_values('score',ascending=False)


    tfadata.uns[f"regulator_df_{branch}"] = b_correlation_df
    tfadata.uns[f"regulator_tf_mat_{branch}"] = b_tf_traj_mat
    tfadata.uns[f"regulator_gene_mat_{branch}"] = b_gene_traj_mat


    #print(list(b_correlation_df.index))
#endf branch_regulator_detect



def mbranch_regulator_detect(adata:AnnData,
                             tfadata:AnnData,
                             branch:Tuple=None,
                             name:str = None,
                             tree = "fate_tree",
                             tree_attr="original",
                             ratio=0.5,
                             intersect_regulator=60,
                             vs_name="auto",
                             log2fc="auto",
                             correlation="auto",
                             ):
    """
    Merged branches to main branches regulators detection.

    Parameters
    ----------
    adata: AnnData
        the data object
    tfadata: AnnData
        the data object of transcription factors
    branch: str
        the end branch name tuple
    tree: str
        the fate tree name
    tree_attr: str
        the fate tree attribute
    ratio: float
        the ratio of cells in the branch to be selected
    intersect_regulator: int
        the number of intersected regulators from differentiation or correlations
    vs_name: str
        the name of the comparison branch default: {tree}_level_{branch}
    log2fc: Union[str, float]
        the log2fc cutoff for differentiation default: "auto"
    correlation: Union[str, float]
        the correlation cutoff for correlation default: "auto"
    """

    if not name:
        raise Exception("Please give a meaningful name!")

    helping_merged_tree(adata, outname=f'{tree}_main')
    helping_merged_tree(tfadata, outname=f'{tree}_main')
    branches = tree_unique_node(tfadata.uns[f'{tree}_main'], 'original')
    branches = [set(i) for i in branches]
    assert(set(branch) in branches)
    branches = [i for i in branches if i != set(branch)]
    branch_2 = [tuple(i) for i in branches]

    print("against branches",branch_2)

    tree_mbranch_markers(tfadata,
                         branches_1=set([branch]),
                         branches_2= set(branch_2),
                         tree_attr=tree_attr,
                         ratio=ratio, vs_name=f'{name}vs',
                         fate_tree=f'{tree}_main',
                         compare_position="end")

    df = get_markers_df(tfadata, f'markers_{name}vs').sort_values("logfoldchanges", ascending=False)

    if log2fc=="auto" or log2fc=="AUTO":
        df_top_names = df.names[:intersect_regulator] ## select 2 times of the differentiation
    else:
        df_top_names = df.names[(df.logfoldchanges > log2fc) & (df.pvals < 0.05)]

    ## calculate the correlation
    b_tf_traj_mat = tree_branches_smooth_window(tfadata, end_branching_node=branch, tree_attr=tree_attr, fate_tree=f'{tree}_main')
    b_gene_traj_mat = tree_branches_smooth_window(adata, end_branching_node=branch, tree_attr=tree_attr, fate_tree=f'{tree}_main')
    b_correlation = branch_TF_gene_correlation(b_tf_traj_mat, b_gene_traj_mat)
    b_correlation_df = pd.DataFrame.from_records(b_correlation, columns =['sym', 'score'], index=[i[0] for i in b_correlation])
    b_correlation_df.sort_values("score", ascending=False, inplace=True)


    if correlation=="auto" or correlation=="AUTO":
        b_top_correlation = b_correlation_df.index[:intersect_regulator] ## select 2 times of the correlation
    else:
        b_top_correlation = [sym for sym, corr in b_correlation if corr > correlation]

    b_inter_TFs   = [i.upper() for i in list(set(b_top_correlation) & set(df_top_names))]
    b_correlation_df = b_correlation_df.loc[b_inter_TFs,:].sort_values('score',ascending=False)


    tfadata.uns[f"regulator_df_{name}"] = b_correlation_df
    tfadata.uns[f"regulator_tf_mat_{name}"] = b_tf_traj_mat
    tfadata.uns[f"regulator_gene_mat_{name}"] = b_gene_traj_mat
#endf mbranch_regulator_detect
