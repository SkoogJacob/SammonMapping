import numpy as np
from models.cdist import cdist


def sammon(x: np.ndarray, max_iter: int = 300, epsilon: float = 1, k: int = 2, maxhalves: int = 16, verbose: int = 0,
           minsize=0.00000000001):
    """
    This function performs sammon mapping to reduce the dimensionality of the data in X
    It will keep iteratively improving the mapping until the cost is below a set threshold, or
    until max_iter iterations have been made.

    To read about sammon mapping, see this wikipedia article: https://en.wikipedia.org/wiki/Sammon_mapping

    :param x: The data to map with sammon mapping. Should be an numpy.ndarray of shape n*p, where n is the number of entries and p is the number of features.
    :param max_iter: The maximum number of iterations for the sammon mapping to try and find the optimal mapping.
    :param epsilon: The cost of the mapping the function strives to get under
    :param k: the dimensionality of the target space. Default is 2.
    :param maxhalves: how many times the algorithm will try to halve the gradient step size if E doesn't decrease
    :param verbose: if greater than 1 will print out E after each iteration
    :param minsize: if values in the distance matrices are smaller than this value, they will be set to this value
    :return: An array of points in the new space of dimensionality q.
    """
    N = x.shape[0]
    y = np.random.randn(N, k)
    D = cdist(x, x)  # Saving all pairwise distances between points in x. Each distance is present twice!
    # equiv to dstar in the original paper
    Dsuminv = 1 / D.sum()  # Summing the pairwise distances. Note each distance is present twice, making the sum be
    # Twice as big as it should be compared to in the paper. In terms of the paper this is
    # 1 / 2c
    D = D + np.eye(N)  # Adding the identity matrix to D as D will be used in division quite a bit where the one is a
    # neutral entity.
    D = np.where(D < minsize, minsize, D)  # Swapping values in D that are too small for minsize
    if np.count_nonzero(D) < D.size:
        print("Warning! x seems to contain identical points!")
    Dinv = 1 / D  # To avoid divisions by 0, take the inverse now and use it in multiplication later
    one = np.ones(y.shape)  # Creating an array of ones for summations later
    d = cdist(y, y) + np.eye(N)  # Taking pairwise distances of y. Equivalent to d in original paper. Adding identity
    d = np.where(d < minsize, minsize, d)
    # To allow for division with it
    dinv = 1 / d  # Same reasoning as for Dinv
    delta = D - d  # For sammon's stress, take distance between the distance

    # Now compute stress
    # Note that when squaring the double size problem also squares, thus dividing it further
    # by 2
    E = 0.5 * Dsuminv * ((delta ** 2) * Dinv).sum()
    current_iter = 0
    while E > epsilon and current_iter < max_iter:
        j, E_new = 0, 0.
        delta = dinv - Dinv  # This equals 1/d[i,j] - 1/D[i,j] = (D[i,j] - d[i,j]) / d[i,j]D[i,j], which is the first
        # Factor in computing the first derivative of E
        deltas = delta.dot(one)  # Creates an array of shape (N, q) where each entry
        # e[i, q] = sum[j, N, j != i] ( (D[i,j] - d[i,j]) / d[i,j]D[i,j] )
        y_diffs = np.zeros((N, k, N))
        first_derivative = -1 * (deltas * y - delta.dot(y))  # Since both first derivative and second are multiplied by
        # -2/c, and the absolute is taken of the second derivative
        # only the sign is important to keep in the first derivative
        y2 = y ** 2
        dinv3 = dinv ** 3
        # Second derivative can be written as "deltas - (1/d^3)(y[i] - y[j])^2"
        # calculating in two steps, starting by taking dinv**3 ( y[p]^2 + y[j]^2) - 2 dinv**3 y[p]y[j] and then
        # adding deltaones, end by taking absolute value
        second_derivative = dinv3.dot(one) * y2 + dinv3.dot(y2) - 2 * y * dinv3.dot(y) - deltas

        step = 0.3 * first_derivative / np.abs(second_derivative)  # step = MF * UpperCaseDelta(m)
        y_old = y

        for j in range(maxhalves):  # Does progressively smaller gradient steps as long as the new step didn't improve
            # stress
            y = y_old - step
            d = cdist(y, y) + np.eye(N)
            d = np.where(d < minsize, minsize, d)
            dinv = 1 / d
            delta = D - d
            E_new = 0.5 * Dsuminv * ((delta ** 2) * Dinv).sum()
            if E_new < E:
                break
            else:
                step = step / 2

        if j == maxhalves - 1:
            print('Warning! Steps seem to be too large, mapping may not converge')
        if E_new >= E:
            print('Warning! Mapping did not improve, returning current mapping!')
            break

        E = E_new
        current_iter = current_iter + 1
        if verbose > 0:
            print(f'Finished iteration {current_iter} with E = {np.around(E, decimals=4)}')
    if current_iter == max_iter:
        print("Warning! max_iter reached, mapping may not have converged.")
    return [y, E]


def sammons_stress(x: np.ndarray, y: np.ndarray, matrixtypes: str = "raw", c=None):
    """
    Computes sammon stress, which is the "cost" of sammon mapping.
    The algorithm compares the distances between points in the high-dimensional input space
    and the low-dimensional output space.
    :param x: The input matrix in high-dimensional space.
    :param y: The output matrix in low-dimensional space.
    :param matrixtypes: The type of matrices that have been passed. Recognised values are 'raw' or 'distance'
    :param c: The c constant from sammons paper. Note that cdist(X,X).sum() = 2*c, so remember to divide by 2 if that is used
    :return: The sammon stress
    """
    # Since cdist takes pairwise mapping, the diagonal of the cdist of the same matrix should be 0
    d_star, d = None, None
    if matrixtypes == 'raw':
        d_star = cdist(x, x)  # Pairwise distances in original data
        d = cdist(y, y)  # Pairwise distances in the reduced dimensionality data
    elif matrixtypes == 'distance':
        d_star, d = x, y

    if np.count_nonzero(np.diagonal(d_star)) > 0:
        raise ValueError("The diagonal of cdist(X,X) should always be 0 (as it is point (x_i-x_i))")
    if np.count_nonzero(np.diagonal(d)) > 0:
        raise ValueError("The diagonal of cdist(Y,Y) should always be 0 (as it is point (y_i-y_i))")

    # Calculating 1 / SUM[i<j](d_star[i,j]. Even though each distance is counted twice, it is not necessary to divide
    # the summation by 2, as one doubling happens in the denominator and one in the enumerator, thus cancelling each
    # other out naturally. Same logic for SUM(d_ij - d_star_ij)^2/d_star_ij) Since division and stuff will be used,
    # adding one to the diagonal to avoid divide by 0
    eye = np.eye(d_star.shape[0])
    if c is None:
        cx2 = d_star.sum()
    else:
        cx2 = c * 2
    diff_square = (d - d_star) ** 2
    d_star_eye = d_star + eye
    if np.count_nonzero(d_star_eye) < d_star_eye.size:
        raise ValueError("Original vector space seems to contain identical data points")
    stress = 0.5 * (1 / cx2) * (diff_square / d_star_eye).sum()
    return stress
