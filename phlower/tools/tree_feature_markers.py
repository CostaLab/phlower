import os
import scipy
import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
from typing import Union, List, Tuple, Dict, Set
from .tree_utils import _edge_two_ends, _edgefreq_to_nodefreq, tree_original_dict
from ..util import get_quantiles


def find_a_branch_all_predecessors(tree:nx.DiGraph=None,
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
    return ret_list[::-1]
#endf find_a_branch_all_predecessors

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

    travesed_nodes = list(nx.dfs_tree(tree, branching_node))
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
                       iscopy = False,
                       vs1_name:str = None,
                       vs2_name:str = None,
                       vs_name:str = None,
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
        if node2 not in adata.uns["fate_tree"].nodes():
            raise ValueError(f"{node2} is not in fate_tree")
    for node1 in nodes1:
        if node1 not in adata.uns["fate_tree"].nodes():
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
            node1_tuple = adata.uns['fate_tree'].nodes[node1]['ecount']
            nodes1_tuple.extend(node1_tuple)
    ## construct node1 array
    else:
        nodes1_tuple = adata.uns['fate_tree'].nodes[nodes1]['ecount']

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
            node2_tuple = adata.uns['fate_tree'].nodes[node2]['ecount']
            nodes2_tuple.extend(node2_tuple)
    ## construct node2 array
    else:
        nodes2_tuple = adata.uns['fate_tree'].nodes[nodes2]['ecount']

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
    print("shapes: ", X_nodes1.shape, X_nodes2.shape)
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
    iscopy: bool
        whether to return a copy of the AnnData object
    kwargs:
        additional arguments passed to tree_nodes_markers

    """
    import functools

    node_groups = _divide_nodes_to_branches(adata.uns['fate_tree'], branching_node)
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
        pre_branch = find_a_branch_all_predecessors(adata.uns['fate_tree'], branching_node)
        len_pre_branch = max(int(len(pre_branch)*ratio), 1)
        node_groups["branching"] = (pre_branch[-1*len_pre_branch:])
        branches_2_predix.append("branching")

    adata = adata.copy() if iscopy else adata

    nodes1 = node_groups[branch_1_predix]
    nodes2 = functools.reduce(lambda x,y: x+y, [node_groups[k] for k in branches_2_predix])


    vs2_name = f"{branching_node}_rest" if include_pre_branch else f"rest"
    if name_append is not None:
        vs2_name = f"{vs2_name}_{name_append}"

    tree_nodes_markers(adata, nodes1, nodes2,  vs1_name=f"{branch_1}", vs2_name=vs2_name,**kwargs)
    return adata if iscopy else None
#endf tree_branches_markers

def tree_mbranch_markers(adata: AnnData,
                         branches_1: set([tuple]),
                         branches_2: set([tuple]),
                         tree_attr:str="original",
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
        the attribute of the tree to be used， default is node_name, could also be original
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
            branch_1_d = tree_original_dict(adata.uns['fate_tree'], branch_1)
            if len(branch_1_d) <1 :
                raise Exception(f"branch_1 is not in the tree")
            branch_1 = find_branch_end(adata.uns['fate_tree'], list(branch_1_d.keys())[0])
        nodes1 = find_a_branch_all_predecessors(adata.uns['fate_tree'], branch_1)
        nodes1_list.extend(nodes1)

    for branch_2 in branches_2:
        if tree_attr == "node_name":
            pass
        elif tree_attr == "original":
            #convert original to node_name
            branch_2_d = tree_original_dict(adata.uns['fate_tree'], branch_2)
            if len(branch_2_d) <1:
                raise Exception(f"branch_1 or branch_2 is not in the tree")
            branch_2 = find_branch_end(adata.uns['fate_tree'], list(branch_2_d.keys())[0])
        nodes2 = find_a_branch_all_predecessors(adata.uns['fate_tree'], branch_2)
        nodes2_list.extend(nodes2)

    nodes1 = nodes1_list
    nodes2 = nodes2_list

#    branching_node1 = find_last_branching(adata.uns['fate_tree'], branch_1)
#    branching_node2 = find_last_branching(adata.uns['fate_tree'], branch_2)
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
#        pre_branch = find_a_branch_all_predecessors(adata.uns['fate_tree'], branching_node1)
#        len_pre_branch = max(int(len(pre_branch)*ratio), 1)
#        nodes_branching = (pre_branch[-1*len_pre_branch:])
#        nodes2 += nodes_branching


    adata = adata.copy() if iscopy else adata
    against_name = f"{branch_2}" if not name_append else f"{branch_2}_{name_append}"
    tree_nodes_markers(adata, nodes1, nodes2,  vs1_name=f"{branch_1}", vs2_name=against_name, vs_name=vs_name, **kwargs)
    return adata if iscopy else None
#endf tree_mbranch_markers


def tree_2branch_markers(adata: AnnData,
                         branch_1: Union[str, tuple] ,
                         branch_2: Union[str, tuple],
                         tree_attr:str="original",
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
        the attribute of the tree to be used， default is node_name, could also be original
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
        branch_1_d = tree_original_dict(adata.uns['fate_tree'], branch_1)
        branch_2_d = tree_original_dict(adata.uns['fate_tree'], branch_2)
        print(branch_1_d)
        print(branch_2_d)
        if len(branch_1_d) <1 or len(branch_2_d) <1:
            raise Exception(f"branch_1 or branch_2 is not in the tree")

        branch_1 = find_branch_end(adata.uns['fate_tree'], list(branch_1_d.keys())[0])
        branch_2 = find_branch_end(adata.uns['fate_tree'], list(branch_2_d.keys())[0])


    print(branch_1, branch_2)
    nodes1 = find_a_branch_all_predecessors(adata.uns['fate_tree'], branch_1)
    nodes2 = find_a_branch_all_predecessors(adata.uns['fate_tree'], branch_2)

    branching_node1 = find_last_branching(adata.uns['fate_tree'], branch_1)
    branching_node2 = find_last_branching(adata.uns['fate_tree'], branch_2)


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
        pre_branch = find_a_branch_all_predecessors(adata.uns['fate_tree'], branching_node1)
        len_pre_branch = max(int(len(pre_branch)*ratio), 1)
        nodes_branching = (pre_branch[-1*len_pre_branch:])
        nodes2 += nodes_branching


    adata = adata.copy() if iscopy else adata
    against_name = f"{branch_2}" if not name_append else f"{branch_2}_{name_append}"
    tree_nodes_markers(adata, nodes1, nodes2,  vs1_name=f"{branch_1}", vs2_name=against_name, vs_name=vs_name, **kwargs)
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
                        tfbdata: AnnData,
                        name:str='markers_3_97_vs_0_11_rest_0.3withParents'):
    """
    calculate the correlation between TF and genes for a comparison of markers
    This is only calculate the correlation without caring about the order of cells


    Parameters
    ----------
    adata: AnnData
        AnnData object with gene expression
    tfbdata: AnnData
        AnnData object with TF binding data
    name: str
        the name of the differentiation branches
    """
    from collections import Counter, defaultdict
    from scipy.stats import pearsonr

    zdata = tfbdata.uns[name].copy()
    group_obs =  zdata.obs_names
    group_obs1 = set([x[:-3] for x in zdata.obs_names if x.endswith('n1')])
    group_obs2 = set([x[:-3] for x in zdata.obs_names if x.endswith('n2')])
    compare_obs = list(group_obs1 | group_obs2)

    TFs = [x[0] for x in zdata.uns['rank_genes_groups']['names']]
    shared_symbol = list(set(adata.var_names) & set(TFs))

    expression_mtx = adata[compare_obs, shared_symbol].X
    TF_mtx = tfbdata[compare_obs, shared_symbol].X

    d_corr = defaultdict(float)
    for idx, sym in enumerate(shared_symbol):
        expression = expression_mtx[:, idx].toarray().ravel()
        TF =  TF_mtx[:, idx].toarray().ravel()
        corr = pearsonr(expression, TF)[0]
        if np.isnan(corr):
            continue
        d_corr[sym] = pearsonr(expression, TF)[0]
    corr_ordered = sorted(d_corr.items(), key=lambda x:x[1], reverse=True)

    return corr_ordered
#endf TF_gene_correlation


def tree_branches_smooth_window(adata: AnnData,
                                start_branching_node: str="",
                                end_branching_node: str="",
                                tree_attr:str="original",
                                fate_tree: str= "fate_tree",
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
        end_branching_node_d = tree_original_dict(adata.uns['fate_tree'], end_branching_node)
        end_branching_node = find_branch_end(adata.uns['fate_tree'], list(end_branching_node_d.keys())[0])
    if not start_branching_node:
        start_branching_node = find_last_branching(adata.uns[fate_tree], end_branching_node)

    if start_branching_node not in adata.uns['fate_tree'].nodes:
        raise ValueError(f"start_branching_node {start_branching_node} is not in the fate tree")
    if end_branching_node not in adata.uns['fate_tree'].nodes:
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
#endf branch_TF_gene_correlation

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
