import copy
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from .util import networkx_node_to_df, kde_eastimate, bfs_layers

def trajectory_buckets(g=None,
                       layouts=None,
                       trajs=[],
                       cluster_list=[],
                       retain_clusters=[],
                       node_attribute='u',
                       sample_n=10000,
                       min_kde_quant_rm=0.1,
                       bucket_number=10):
    """
    Bucket the trajectory clusters into a number of buckets.
    """

    if len(retain_clusters) == 0:
        retain_clusters = set(cluster_list)
    if len(cluster_list)==0:
        print("Error: cluster_list should not be NULL!")
        return

    assert(set(retain_clusters).issubset(set(cluster_list)))
    assert(len(trajs) == len(cluster_list))

    buckets_df_dict = {}
    for cluster in tqdm(retain_clusters):
        idx = [i for i in np.where(np.array(cluster_list) == cluster)[0]]
        kde = kde_eastimate(np.array(trajs, dtype=list)[idx], layouts, sample_n=sample_n)
        ## left merge to keep kde index
        df_left = pd.merge(pd.DataFrame(kde), networkx_node_to_df(g), how='left', left_on = 'idx', right_on = 'node')
        ## remove the left quantile of the kde
        df_left = df_left[df_left.z >df_left.z.quantile(min_kde_quant_rm)]

        ## buckets by the node attribute rank
        df_left['rank'] = df_left[node_attribute].rank()
        bins = np.linspace(df_left['rank'].min(), df_left['rank'].max(), bucket_number+1)
        df_left['buckets'] = pd.cut(df_left['rank'], bins , include_lowest=True)
        d_buckets = {v:i for  i, v in enumerate(sorted(set(df_left.buckets)))}

        df_left[bucket_idx_name] = df_left['buckets'].apply(lambda x: d_buckets[x])

        buckets_df_dict[cluster] = df_left

    return buckets_df_dict



def initial_a_tree(df_1, df_2, name_1, name_2, bucket_idx_name='buckets_idx',  intersect_ratio=0.4,  decay_ratio=1.05):
    """
    Create a tree from two pseudotime buckets dataframe
    Two pseudotime buckets lists, move a step see if we need branching.
    If the two buckets have a large intersection with one more step than the other, we slow down the one with more intersection.


    Parameters
    ----------
    df_1: pd.DataFrame, dataframe stores the pseudotime buckcket indexs
    df_2: pd.DataFrame, dataframe stores the pseudotime buckcket indexs
    name_1: str
    name_2: str
    bucket_idx_name: str
    intersect_ratio: float, if the intersection of two buckets is smaller than this ratio, the tree will branching
    decay_ratio: float: the later the pseudotime, the more likely the tree will branching
    """
    cursor_1 = 0
    cursor_2 = 0

    bucket_max_1 = max(set(df_1[bucket_idx_name][pd.notna(df_1[bucket_idx_name])]))
    bucket_max_2 = max(set(df_2[bucket_idx_name][pd.notna(df_2[bucket_idx_name])]))
    a = set(df_1[df_1[bucket_idx_name] == cursor_1].idx)
    b = set(df_2[df_2[bucket_idx_name] == cursor_2].idx)
    print(bucket_max_1,bucket_max_2)
    m_i = 0
    graph = nx.DiGraph()
    #print(a, b)
    graph.add_node(0,  who={name_1, name_2}, cells=a|b)
    while cursor_1 <= bucket_max_1 and cursor_2 <= bucket_max_2:
        a = set(df_1[df_1[bucket_idx_name] == cursor_1].idx)
        b = set(df_2[df_2[bucket_idx_name] == cursor_2].idx)
        a1 =set(df_1[df_1[bucket_idx_name] == cursor_1+1].idx) ## look ahead one step
        b1 =set(df_2[df_2[bucket_idx_name] == cursor_2+1].idx) ## look ahead one step
        intersect_ratio = intersect_ratio * decay_ratio
        min_len = min(len(a), len(b)) * intersect_ratio ## condition if we need branching
        a1b1 = len(a1&b1)
        a1b = len(a1 & b)
        ab1 = len(a & b1)

        if a1b1 > a1b and a1b1 > ab1:
            cursor_1 += 1
            cursor_2 += 1
            if a1b1 > min_len:
                m_i += 1
                if not graph.has_node(m_i):
                    graph.add_node(m_i, who={name_1, name_2}, cells=a1|b1)
                else:
                    graph.nodes[m_i]["cells"] = graph.nodes[m_i]["cells"] | a1|b1
                if m_i > 0:
                    graph.add_edge(m_i-1, m_i)
            else:
                #print("a1b1")
                break
        elif a1b > a1b1 and a1b >a1b1:
            cursor_1 +=1
            if a1b > min_len:
                if not graph.has_node(m_i):
                    graph.add_node(m_i,  who={name_1, name_2}, cells=set(a1 | b))
                else:
                    graph.nodes[m_i]["cells"] = graph.nodes[m_i]["cells"] | a1 | b
                if m_i > 0:
                    graph.add_edge(m_i-1, m_i)

            else:
                break
        elif ab1 > a1b1 and ab1 > a1b1:
            cursor_2 +=1
            if ab1 > min_len:
                if not graph.has_node(m_i):
                    graph.add_node(m_i, who={name_1, name_2},  cells=set(a|b1))
                else:
                    graph.nodes[m_i]["cells"] = graph.nodes[m_i]["cells"] | a | b1
                if m_i > 0:
                    graph.add_edge(m_i-1, m_i)
            else:
                #print("ab1")
                break
        #print(cursor_1, cursor_2, m_i)

    for i in range(cursor_1, bucket_max_1+1):
        graph.add_node(f"{name_1}_{i}", who={name_1}, cells = set(df_1[df_1[bucket_idx_name] == i].idx))
        #print(i)
        if i==cursor_1:
            graph.add_edge(m_i, f"{name_1}_{i}")
        else:
            graph.add_edge(f"{name_1}_{i-1}", f"{name_1}_{i}")

    for i in range(cursor_2, bucket_max_2+1):
        graph.add_node(f"{name_2}_{i}", who={name_2}, cells=set(df_2[df_2[bucket_idx_name] == i].idx))
        #print(i)
        if i==cursor_2:
            graph.add_edge(m_i, f"{name_2}_{i}")
        else:
            graph.add_edge(f"{name_2}_{i-1}", f"{name_2}_{i}")
    return graph



def add_traj_to_graph(graph, df, name, bucket_idx_name='buckets_idx', intersect_ratio=0.4, ratio_decay=1.05):
    graph_out = copy.deepcopy(graph)
    bucket_max_b = max(set(df[bucket_idx_name][pd.notna(df[bucket_idx_name])]))
    offset_b = 0
    layers = list(bfs_layers(graph_out, 0))
    branching_point = ""
    for idx, alist in enumerate(layers):
        node = alist[0]
        if idx == 0:
            a = graph_out.nodes[node]['cells']
            b = set(df[df[bucket_idx_name] == idx].idx)
            #min_len = min(len(a), len(b)) * intersect_ratio
            graph_out.nodes[node]['cells'] = a | b
            continue

        if len(alist) > 1:
            a_nodes = [graph_out.nodes[node]['cells'] for node in alist]
            b = set(df[df[bucket_idx_name] ==  idx+offset_b].idx)
            max_idx = np.argmax(np.array([len(a & b) for a in a_nodes]))
            #for a in a_nodes:
            #    print('node', , len(a), len(b))

            a = a_nodes[max_idx]
            print("branching", alist[max_idx])
            intersect_ratio = intersect_ratio * ratio_decay
            min_len = min(len(a), len(b)) * intersect_ratio

            ## this case, we along one trajectory and merge to the end.
            if len(a & b) > min_len:
                ## dfs traval the tree from this point
                dfs_nodes = list(nx.dfs_preorder_nodes(graph_out, alist[max_idx]))
                for dfs_idx, node in enumerate(dfs_nodes):
                    a = graph_out.nodes[node]['cells']
                    b = set(df[df[bucket_idx_name] == idx+offset_b+dfs_idx].idx)
                    #print("handling: ", idx+offset_b + dfs_idx)
                    if len(a & b) > min_len:
                        print("add ", node, " ", idx+ offset_b+dfs_idx)
                        graph_out.nodes[node]['cells'] = a | b
                        graph_out.nodes[node]['who'] = graph_out.nodes[node]['who'] | {name}
                    else:
                        print("branching: ", idx+offset_b+dfs_idx)
                        print(dfs_idx)
                        branching_point = dfs_nodes[dfs_idx - 1]
                        break
                    intersect_ratio = intersect_ratio * ratio_decay
                    min_len = min(len(a), len(b)) * intersect_ratio
                for i in range(idx + offset_b + dfs_idx, bucket_max_b+1):
                    graph_out.add_node(f"{name}_{i}", who={name}, cells=set(df[df[bucket_idx_name] == i].idx))
                    if i==idx+offset_b + dfs_idx:
                        graph_out.add_edge(branching_point, f"{name}_{i}")
                    else:
                        graph_out.add_edge(f"{name}_{i-1}", f"{name}_{i}")
                branching_point = ""
            else:
                branching_point = layers[idx-1][0]

            break
            #if idx+offset_b+1 > bucket_max_b:
            #    break
        else:
            a = graph_out.nodes[node]['cells']
            b = set(df[df[bucket_idx_name] == idx+offset_b].idx)
            b1 = set(df[df[bucket_idx_name] == idx+offset_b+1].idx)

            intersect_ratio = intersect_ratio * ratio_decay
            min_len = min(len(a), len(b)) * intersect_ratio
            if len(a & b) > min_len:
                print("add ", node, " ", idx+ offset_b)
                graph_out.nodes[node]['cells'] = a | b
                graph_out.nodes[node]['who'] = graph_out.nodes[node]['who'] | {name}
            else: ## start a new branch.
                branching_point = layers[idx - 1][0]
                break

            ## if next node is also likely merge into this node.
            if len(a & b1) > len(a & b):
                print("add ", node, " ", idx+ offset_b + 1)
                #print('offset + 1 =======================================================')
                offset_b = offset_b + 1
                graph_out.nodes[node]['cells'] = graph_out.nodes[node]['cells'] | b1

    if branching_point:
        for i in range(idx+offset_b, bucket_max_b+1):
            print(i)
            graph_out.add_node(f"{name}_{i}", who={name}, cells=set(df[df[bucket_idx_name] == i].idx))
            #print(i)
            if i==idx+offset_b:
                graph_out.add_edge(branching_point, f"{name}_{i}")
            else:
                graph_out.add_edge(f"{name}_{i-1}", f"{name}_{i}")


    return graph_out
