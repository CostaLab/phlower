from scipy.spatial import distance
from sklearn.metrics import auc, precision_recall_curve, roc_curve, silhouette_score
import sklearn.cluster as skcluster
import numpy as np

def kmeans(lst, n=2):
    km = skcluster.KMeans(n_clusters=n, random_state=0).fit(lst)
    return(km.labels_)

def cluster_aupr(clusters, d_list):
      n  = len(clusters)
      cl = list(set(clusters))
      pred = distance.pdist(clusters.reshape((n,1)), metric="hamming")
      d_list = (d_list - min(d_list)) / (max(d_list) - min(d_list))
      precision, recall, _ = precision_recall_curve(probas_pred=1-d_list, y_true= 1-pred)
      aupri = auc(recall, precision)
      return aupri

def cluster_auc(clusters, d_list):
      n  = len(clusters)
      cl = list(set(clusters))
      pred = distance.pdist(clusters.reshape((n,1)), metric="hamming")
      d_list = (d_list - min(d_list)) / (max(d_list) - min(d_list))
      fpr, tpr, _ =  roc_curve(y_score=1-d_list, y_true=pred)
      auci = auc(tpr, fpr)
      return auci

def cluster_silh(clusters, d_list):
    silh  = silhouette_score(distance.squareform(d_list), clusters)
    return silh


def batch_kmeans_evaluate(mat_coor_flatten_trajectories, krange=range(2,20), verbose=True):
    d_list = distance.pdist(mat_coor_flatten_trajectories)
    list_c = []
    list_pr = []
    list_sil = []
    cluster_dict = {}
    for k in krange:
        clusters = kmeans(mat_coor_flatten_trajectories, k)
        cluster_dict[k] = clusters
        #print(len(clusters))
        c  = cluster_auc(clusters, d_list)
        pr = cluster_aupr(clusters, d_list)
        sil = cluster_silh(clusters, d_list)
        list_c.append(c)
        list_pr.append(pr)
        list_sil.append(sil)
        if verbose:
            print(k, c, pr, sil)
    return {"cluster_dict":cluster_dict, "list_auc":list_c, "list_aupr":list_pr, "list_sil":list_sil}
