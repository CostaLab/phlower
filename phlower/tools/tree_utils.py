from anndata import AnnData
from typing import Union, Optional, Sequence, Tuple, Mapping, Iterable, Callable
import numpy as np
import pandas as pd
import networkx as nx
from ..util import networkx_node_to_df

def remove_duplicated_index(elements):
    """
    return unique index
    """
    index_dic = pd.DataFrame(elements).groupby([0]).indices
    return [v[0] for k,v in index_dic.items()]
#endf remove_duplicated_index

def flatten_tuple(nested_tuple):
    # check if tuple is empty
    if not(bool(nested_tuple)):
        return nested_tuple

     # to check instance of tuple is empty or not
    if isinstance(nested_tuple[0], tuple):

        # call function with subtuple as argument
        return flatten_tuple(*nested_tuple[:1]) + flatten_tuple(nested_tuple[1:])

    # call function with subtuple as argument
    return nested_tuple[:1] + flatten_tuple(nested_tuple[1:])


def TF_to_genes(TFs, ones=False):
    """
    convert TFs to genes for JASPAR database
    if ones is True, return dict of tuples
    if ones is False return dict of strings

    some strange format a(var.2) or a::b::c

    Parameters
    ----------
    TFs : list
        Transcription factors in list
    ones : bool, optional
        If True, return dict of tuples, by default False, else return dict of strings.
    """
    import re
    d = {}
    if ones:
        for TF in TFs:
            gene = re.sub("\\(.*?\\)", "", TF) ## a(var.2) -> a
            gene1 = re.sub("(.*)::(.*)", "\\1", gene) ## a::b::c -> a::b
            gene2 = re.sub("(.*)::(.*)", "\\2", gene1) if "::" in gene1 else "" ## a::b -> b
            gene3 = re.sub("(.*)::(.*)", "\\2", gene) if "::" in gene else "" ## a::b::c -> c
            gene1 = gene1 if "::" not in gene1 else re.sub("(.*)::(.*)", "\\1", gene1)## gene1 a::b--> a
            gene1 = re.sub("(.*)-(.*)", "\\1", gene1) ## a-b -> a
            d[TF] = tuple(i for i in [gene1, gene2, gene3] if i)
    else:
        for TF in TFs:
            gene = re.sub("\\(.*?\\)", "", TF) ## a(var.2) -> a
            gene = re.sub("(.*)::(.*)", "\\1", gene) ## a::b::c -> a::b
            gene = re.sub("(.*)::(.*)", "\\1", gene) ## a::b -> a
            gene = re.sub("(.*)-(.*)", "\\1", gene) ## a-b -> a
            d[TF] = gene
    return d
#endf TF_to_genes

def _edgefreq_to_nodefreq(edge_freq, d_edge2node):
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


def _edge_two_ends(adata: AnnData,
                   graph_name: str = None,
                   ):

    if "graph_basis" in adata.uns.keys() and not graph_name:
        graph_name = adata.uns["graph_basis"] + "_triangulation_circle"

    elist = np.array([(x[0], x[1]) for x in adata.uns[graph_name].edges()])
    return {i:v for i, v in enumerate(elist)}

def print_stream_labels(adata, tree='stream_tree', attr='original'):
    import functools
    import networkx as nx
    arr_tuple = np.fromiter(nx.get_node_attributes(adata.uns[tree], attr).values(), tuple)
    return functools.reduce(lambda a,b: set(a)|set(b), arr_tuple)



def change_stream_labels(adata, tree='stream_tree', attr='original', from_to_dict={}, iscopy=False):
    """
    Change the labels of the stream tree nodes.
    When annotation is changed, need to change the attribute of the tree nodes.
    """
    adata = adata.copy() if iscopy else adata
    nodes = adata.uns[tree].nodes()
    print("changing:", from_to_dict)
    for node in nodes:
        assert(isinstance(adata.uns[tree].nodes[node][attr], tuple))
        adata.uns[tree].nodes[node][attr] = tuple([from_to_dict.get(i, i) for i in adata.uns[tree].nodes[node][attr]])
    return adata if iscopy else None

def change_fate_labels(adata, tree='fate_tree', attr='original', from_to_dict={}, iscopy=False):
    """
    change not only leaves but also internal nodes
    """
    adata = adata.copy() if iscopy else adata
    nodes = adata.uns[tree].nodes()
    print("changing:", from_to_dict)
    for node in nodes:
        assert(isinstance(adata.uns[tree].nodes[node][attr], tuple))
        a,b = adata.uns[tree].nodes[node][attr]

        for k,v in from_to_dict.items():
            if k in a:
                a = [i if i!=k else v for i in a]
        new_attr = (tuple(a),b)
        adata.uns[tree].nodes[node][attr] = new_attr
    return adata if iscopy else None

#endf change_fate_labels


def get_minimum_nodes(adata, tree='fate_tree', name='4_67'):
    """
    for fate_tree, each branch has a series of nodes increasingly ordered,
    the minimum nodes are the first nodes of each branch
    """

    if name =='root':
        return 'root'

    nodes = adata.uns[tree].nodes()
    predix = name.split('_')[0]
    nodes = [node for node in nodes if node.startswith(predix)]
    appendix = min([int(node.split('_')[1]) for node in nodes])
    return predix + '_' + str(appendix)

def get_maximum_nodes(adata, tree='fate_tree', name='4_67'):
    """
    for fate_tree, each branch has a series of nodes increasingly ordered,
    the maximum nodes are the last nodes of each branch
    """

    if name =='root':
        return 'root'

    nodes = adata.uns[tree].nodes()
    predix = name.split('_')[0]
    nodes = [node for node in nodes if node.startswith(predix)]
    appendix = max([int(node.split('_')[1]) for node in nodes])
    return predix + '_' + str(appendix)
#endf get_maximum_nodes


def get_rank_genes_group(adata, name):
    """
    helping function to get the rank genes of a group from the fate_tree differential expression
    """
    if name not in adata.uns.keys():
        raise ValueError("name not in adata.uns.keys()")
    return adata.uns[name].uns['rank_genes_groups']

def get_markers_df(adata, name):
    rank_groups = adata.uns[name].uns['rank_genes_groups']
    df = pd.DataFrame({"names" : [i[0] for i in rank_groups['names'].tolist()],
                        "scores" : [i[0] for i in rank_groups['scores'].tolist()],
                        "pvals"  : [i[0] for i in rank_groups['pvals'].tolist()],
                        "pvals_adj" : [i[0] for i in rank_groups['pvals_adj'].tolist()],
                        "logfoldchanges" : [i[0] for i in rank_groups['logfoldchanges'].tolist()], }
                      )
    return df

def fate_tree_full_dataframe(adata, tree='fate_tree', graph_name=None):
    dff = networkx_node_to_df(adata.uns[tree])
    d_edge2node = _edge_two_ends(adata)
    dff['ncount'] = dff['ecount'].apply(lambda x: list(_edgefreq_to_nodefreq(x, d_edge2node).items()))
    return dff

def assign_graph_node_attr_to_adata(adata, graph_name='X_pca_ddhodge_g', from_attr='pos', to_attr='pos', iscopy=False):
    """
    assign the node attributes of a graph to adata.obs

    Parameters
    ----------
    adata: AnnData
        The AnnData object
    graph_name: str
        The name of the graph in adata.uns
    from_attr: str
        The attribute of the graph nodes
    to_attr: str
        The attribute of the adata.obs

    Returns
    -------
    AnnData
        The AnnData object with the new attribute
    """
    adata = adata.copy() if iscopy else adata

    if graph_name not in adata.uns.keys():
        raise ValueError("graph_name not in adata.uns.keys()")
    if from_attr not in adata.uns[graph_name].nodes[0].keys():
        raise ValueError("from_attr not in adata.uns[graph_name].nodes[0].keys()")
    if to_attr in adata.obsm.keys():
        print("Warning: to attr already in adata.obsm.keys(), will be overwritten")
    attr = [i[1] for i in sorted(nx.get_node_attributes(adata.uns[graph_name], from_attr).items(), key=lambda x: x[0])]
    adata.obs[to_attr] = attr

    return adata if iscopy else None

#endf assign_graph_node_attr_to_adata

def get_tree_leaves_attr(tree: nx.DiGraph, attr: str = 'original'):
    """
    the implementation of fate tree merge is using the tuple for each node, thus, the tuple lengths are only 1 for the leaf nodes.
    we just extract the name of the leaf nodes as string
    return dict: {node_name: leaf attribute with string or int format}
    """
    leaves = [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1]

    ret_dict = {}
    for leaf in leaves:
        attr_val = tree.nodes[leaf][attr]
        if isinstance(attr_val, tuple):
            attr_val = attr_val[0]
        if isinstance(attr_val, tuple):
            attr_val = attr_val[0]
        if isinstance(attr_val, tuple):
            attr_val = attr_val[0]

        ret_dict[leaf] = attr_val

    #{leaf: tree.nodes[leaf][attr][0][0] if  isinstance(tree.nodes[leaf][attr][0],  tuple) else  tree.nodes[leaf][attr][0] for leaf in leaves }
    return ret_dict
#endf get_tree_leaves_attr


def get_all_attr_names(tree):
    """
    return all the attribute names of the tree nodes
    """
    first_node = list(tree.nodes())[0]
    return list(tree.nodes[first_node].keys())
#endf get_all_attr_names


def tree_label_dict(adata,
                    tree = "fate_tree",
                    from_ = "node_name",
                    to_ = 'original',
                    branch_label = False,
                    ):
    """
    stream_tree: from_&to_ candidate: node_name, original, label
    fate_tree: from_&to_ candidate: node_name, original
    """
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
#endf tree_label_dict


def tree_original_dict(tree, leaf_name):
    """
    Given a node original name like '27-PODO' or ('27-PODO',) or ('27-PODO', "9-PT/LOH"),
    find all the original names in the path return a dict {node_name: original_name}

    Parameters
    ----------
    tree: networkx.DiGraph
        fate tree
    leaf_name: str or tuple
        the original name of the leaf node
    """
    attrs = nx.get_node_attributes(tree, 'original')
    leaf_type = "tuple" if isinstance(leaf_name, tuple) else "str"
    d = {}
    if leaf_type == "str":
        for k,v in attrs.items():
            if len(v[0]) > 1:
                continue
            if v[0][0] == leaf_name:
                d[k] = v
    elif leaf_type == "tuple":
        for k,v in attrs.items():
            if set(v[0]) == set(leaf_name):
                d[k] = v
            elif set(flatten_tuple(v[0])) == set(flatten_tuple(leaf_name)):
                d[k] = v
    return d
#endf tree_original_dict




def to_root_list(tree, node):
    """
    Given a node, return the list of nodes from the node to the root
    """
    return list(nx.shortest_path(tree,'root', node))
#endf to_root_list


def end_branch_dict(adata, branch_id_alias='branch_id_alias', fate_tree='stream_tree', from_='label', to_='original'):
    """
    for adata.obs.branch_id_alias, return a dict {branch_id_alias: original_name}
    like:
      {('S15', 'S14'): 'Stromal-4',
       ('S10', 'S7'): 'Tubular',
       ('S11', 'S8'): 'Stromal-1',
       ('S13', 'S12'): 'Stromal-2',
       ('S9', 'S7'): 'Podocytes',
       ('S5', 'S2'): 'Neuron-3',
       ('S16', 'S14'): 'Stromal-3',
       ('S6', 'S3'): 'root',
       ('S1', 'S0'): 'Neuron-1',
       ('S4', 'S2'): 'Neuron-2'}
    """
    from collections import Counter
    keys=Counter(adata.obs[branch_id_alias])# alias key ('S7', 'S3')
    dic_ = tree_label_dict(adata, fate_tree, from_=from_, to_=to_)# "S15": 'Stromal-4'
    dic_ret = {}
    for k,v in dic_.items():
        if len(v) == 0:
            continue
        for key_tuple in keys:
            if k in key_tuple:
                dic_ret[key_tuple] = v
    return dic_ret



def tree_unique_node(fate_tree, attr='fate_tree'):
    """
    for a fate_tree, ignore the number in each branch of the tree
    only keep the unique branch name in the original attribute
    """
    all_nodes = nx.get_node_attributes(fate_tree, attr)
    s = set()
    for i in all_nodes.values():
        a,b = i
        s.add(a)
    #print(s)
    s.remove(('root',))
    lst = list(s)
    idx =  np.argmax([len(i) for i in lst])
    #lst[3]
    assert(all([set(i)<=set(lst[idx]) for i in lst]))
    del lst[idx] ## remove the
    return lst
