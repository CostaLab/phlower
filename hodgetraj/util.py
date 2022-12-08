import itertools
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def lexsort_rows(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return array[np.lexsort(np.rot90(array))]

def tuple_increase(a,b):
    if a < b:
        return (a,b)
    return(b,a)
#endf tuple_increase

def norm01(a):
    maxx= max(a)
    minn = min(a)
    a = [maxx if np.isnan(i) else i for i in a]
    a = [(i - minn)/(maxx-minn) for i in a]
    return a
#endf norm01

def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
#endf pairwise

def top_n_from(node_arr, score_dict, n, largest=True):
    """
    Select top n largest or smallest nodes from node_arr according the score_dict

    Parameters
    -------
    node_arr: nodes candidates
    score_dict: score dictionary for all nodes
    n: top n
    largest: largest n if True else smallest
    """
    assert(n > 0)
    if isinstance(node_arr, list):
        node_arr = np.array(node_arr)
    score_arr = np.array([*score_dict.values()])[node_arr]
    if largest:
        node_idx = node_arr[np.argpartition(score_arr, -n)[-n:]]
        return node_idx
    else:
        node_idx = node_arr[np.argpartition(score_arr, n)[:n]]
        return node_idx

#endf top_n_from

def is_in_2sets(a,b, set_list):
    """
    if a, b pair in different set_list return True
    else return False
    """
    idx_as = [i for i in range(len(set_list)) if a in set_list[i]]
    idx_bs = [i for i in range(len(set_list)) if b in set_list[i]]
    if (not idx_as) or (not idx_bs):
        return False
    idx_a = idx_as[0]
    idx_b = idx_bs[0]

    if idx_a == idx_b:
        return False
    return True
#endf is_in_2sets


def kde_eastimate(trajectories, layouts, sample_n=4000, seeds=2022):
    """
    run gaussian_kde on trajectory nodes

    Parameters
    -----------
    trajectories: trajectories list
    layouts: x,y coordinate of each point
    sample_n: sample to reduce the computing time
    seeds: seeds for sampling
    """

    xy = [layouts[a] for t in trajectories for a in t]
    x = np.array([x for x, y in xy])
    y = np.array([y for x, y in xy])

    # Calculate the point density
    xy = np.vstack([x,y])

    np.random.seed(seeds)
    sub_idx = np.random.choice(range(xy.shape[1]), min(sample_n, len(x)), replace=False)
    xy = xy[:, sub_idx]
    x = x[sub_idx]
    y = y[sub_idx]
    z = gaussian_kde(xy)(xy)
    index = np.array([a for t in trajectories for a in t])[sub_idx]

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    index = index[idx]

    return {'idx':index, 'x':x, 'y':y, 'z':z}
#endf

def intersect_kde(traj_dicts, ratio=0.8):
    """
    use the density information to get the insert branching points

    Parameters
    ----------
    traj_dicts: trajectory density dictionary, key is each trajectory class, value is the density(x,y,z,idx)
    ratio: ratio to keep for the intersection

    Return
    ----------
    intersection of all nodes.
    """
    df_list = [(pd.DataFrame(traj)).drop_duplicates() for traj in traj_dicts.values()]
    set_list = [set((df.nlargest(int(len(df) *ratio), 'z'))['idx']) for df in df_list]
    return set.intersection(*set_list)
#endf
