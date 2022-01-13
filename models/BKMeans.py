import numpy as np
from sklearn.cluster import KMeans


def bkmeans(X: np.ndarray, k: int, iter: int):
    """
    This function uses a bisecting k-Means strategy for creating clusters from a dataset.
    It uses a topdown approach, starting with all of the passed data and then dividing it into smaller clusters.
    It keeps bisecting the larger cluster until there are k clusters. For each bisection it tries k-means to find the
    correct bisection iter times.

    :param X: The data, which should be an n*p array, where n is the number of entries and p the number of features.
    :param k: How many clusters the data should be divided into
    :param iter: How many iterations the function should try to find the correct bisection.
    :return: An ndarray with shape (n,), where each entry has a number between 0 and k-1, and the value of the entry indicates
    what cluster the data was put into.
    """
    clustering = None
    if k > 1:
        clustering = bisect_cluster(X, iter)
    else:
        return np.zeros(X.shape[0])
    for clusterCount in range(1, k):
        largestClusterIndex = 0
        largestCluster = 0
        for cindex in range(clusterCount):
            csize = X[clustering == cindex, :].size
            if csize > largestCluster:
                largestCluster = csize
                largestClusterIndex = cindex
        subcluster = bisect_cluster(X[clustering == largestClusterIndex], iter)
        subcluster = np.where(subcluster == 0, largestClusterIndex,
                              cindex + 1)  # Let the lower cluster keep old cluster value, the other gets to be the "new" cluster
        clustering[clustering == largestClusterIndex] = subcluster
    return clustering


def bisect_cluster(X: np.ndarray, iter: int):
    bisector = KMeans(n_clusters=2, n_init=iter)
    clustering = bisector.fit_predict(X)
    return clustering
