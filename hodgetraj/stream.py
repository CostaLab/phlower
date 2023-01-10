import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from .util import networkx_node_to_df, kde_eastimate

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

        df_left['buckets_idx'] = df_left['buckets'].apply(lambda x: d_buckets[x])

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
