import numpy as np
import scipy.spatial.distance


def cdist(XA: np.ndarray, XB: np.ndarray, metric='euclidean'):
    """
    Calculates pairwise distances between the two passed arrays.
    If the returned distance array is called D, then D[i,j] will contain the distance between point i in XA and point j in XB
    Thus, if matrix XA has n entries and matrix XB has m entries, the returned matrix will be of shape (n,m)

    This function does the same thing as scipy.spatial.distance.cdist does when two arrays are passed and the rest is left with standard arguments,
    i.e. this is a very limited version of it. If another metric than euclidean is passed, the function will try to call
    scipy.spatial.distance.cdist.

    :param XA: The first matrix of points
    :param XB: The second matrix of points
    :return: A matrix of pairwise distances
    """
    sa, sb = XA.shape, XB.shape
    if len(sa) != 2 or len(sb) != 2:
        raise ValueError("arrays must be 2-dimensional!")
    if sa[1] != sb[1]:
        raise ValueError("Arrays must have same number of rows (i.e. be in spaces of the same dimensionality)")

    if metric != 'euclidean':
        return scipy.spatial.distance.cdist(XA, XB, metric=metric)

    dm = np.zeros((sa[0], sb[0]))  # Distance matrix
    for i in range(sa[0]):
        for j in range(sb[0]):
            dm[i, j] = np.linalg.norm(XA[i] - XB[j])
    return dm


if __name__ == '__main__':
    X = np.random.rand(6, 5)
    Y = np.random.rand(3, 5)
    d = cdist(X, Y)
    print(d.shape)
    print(d[0])

    xd = cdist(X, X)
    yd = cdist(Y, Y)

    print(xd)
    print(yd)
