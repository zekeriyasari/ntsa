import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import argrelextrema


def counts2pmf(counts):
    """
    Calculate the probablity mass function(pmf) from counts
    :param counts: np.ndarray,
        number of counts
    :return: np.ndarray,
        pmf of random variable x
    """
    pmf = counts / np.sum(counts)
    return pmf


def entropy(px):
    """
    Entropy of time series x with probability mass function (pmf) px
    :param px: np.ndarray,
        pmf of x
    :return: float,
        entropy of x
    """
    assert (0 <= px.all() <= 1), 'Probabilities must be in the interval [0, 1]'
    px = px[px > 0]
    h = -np.sum(px * np.log2(px))
    return h


def mutual_information(x, y, num_bins=100):
    """
    Calculate the mutual information between vectors
    :param x: np.ndarray,
        first time series
    :param y: np.ndarray,
        second time series
    :param num_bins: int,
        number of histogram bins
    :return: float,
        mutual information
    """
    # Calculate the number of occurrences
    x_count = np.histogram(x, bins=num_bins)[0]
    y_count = np.histogram(y, bins=num_bins)[0]
    xy_count = np.histogram2d(x, y, bins=num_bins)[0]

    # Calculate the pmfs
    px = counts2pmf(x_count)
    py = counts2pmf(y_count)
    pxy = counts2pmf(xy_count)

    # Calculate the entropies
    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(pxy)

    # Calculate the mutual information
    mi = hx + hy - hxy
    return mi


def optimum_delay(data, num_lags=50):
    """
    Determine optimum lag for the delay embedding.
    Optimum lag is found to be the first minimum of mutual information.

    :param data:np.ndarray,
        time series
    :param num_lags: int,
        number of lags
    :return: int,
        optimum lag for delay embedding

    References
    ----------
    Abarbanel, Henry. Analysis of observed chaotic data. Springer Science & Business Media, 2012.
    """
    mi = np.array([])
    for l in range(1, num_lags):
        a = data[:-l]
        b = data[l:]
        mi = np.append(mi, mutual_information(a, b))
    opt_lag = argrelextrema(mi, np.less)
    return opt_lag[0][0] + 1


def delay_embedding(data, dim, lag=1):
    """
    Delay embedding
    :param data: np.ndarray,
        scalar data vector
    :param dim: int,
        embedding dimension
    :param lag: int,
        lag
    :return: np.ndarray,
        x = [x_0, \ldots, x_{M}].l,
        where x_i = [s_i, \ldots, s_{i + j * d}, \ldots, s_{i + (dim - 1) * lag}].l
    """
    n = len(data)  # data length
    if n < (dim - 1) * lag + 1:
        raise ValueError('Expected data length for dim={} and lag={} is greater than {}, got {}'.
                         format(dim, lag, (dim - 1) * lag, n))
    p = n - (dim - 1) * lag
    x = np.zeros((p, dim))
    for i in range(p):
        x[i] = np.array([data[i + j * lag] for j in range(dim)])
    return x


def nearest_neighbors(orbit, neigh=2, get_neighbors=False):
    """
    Finds the nearest neighbors index for each row vector of data.
    :param orbit: np.ndarray,
        data matrix for which nearest neighbours are to be found for each row
    :param neigh: int,
        neighborhood distance
    :param get_neighbors: bool,
        if true, return nearest neighbors as well as indexes.
    :return: np.ndarray,

    """
    orbit_nn = np.zeros(orbit.shape)
    tree = cKDTree(orbit)  # construct orbit tree
    row_num = orbit.shape[0]
    indexes = np.zeros(row_num, dtype=int)
    for i in range(row_num):
        index = tree.query(orbit[i], k=neigh)[1][1]
        orbit_nn[i] = orbit[index]
        indexes[i] = index

    if get_neighbors:
        return indexes, orbit_nn
    else:
        return indexes


def distance(x, y, order=2, axis=1):
    """
    Calculate the distance between each row of x and y matrices
    :param x: np.array,
    :param y: np.array
    :param order: int,
        order of norm
    :return: float,
        distance between x and y
    """
    return np.linalg.norm(x - y, ord=order, axis=axis)


def embedding_dimension(data, lag, dist_tol=15, max_dim=10, get_fnn=False):
    """
    Find embedding dimension for the time series data
    :param data: np.ndarray,
        time series
    :param lag: int,
        amount of delay for delay embedding
    :param dist_tol: int,
        threshold for the distance between the nearest neighbours
    :param max_dim: int,
        maximum dimension to search for
    :param get_fnn: bool,
        if true, return all false nearest neighbor ratios.
    :return: int,
        embedding dimension for delay delay embedding.
    """
    n = len(data)
    fnn = []
    for dim in range(1, max_dim):
        # Construct the orbit an find the nearest neighbors
        x = delay_embedding(data, dim, lag=lag)
        x = x[:n - 2 * dim * lag]
        indexes, x_nn = nearest_neighbors(x, get_neighbors=True)

        # Calculate the distances between nearest neighbors
        r = distance(x, x_nn)

        # Calculate the components to be added in dimension d + 1
        vec1 = np.array([data[j + dim * lag] for j in range(x.shape[0])])
        vec2 = np.array([data[indexes[j] + dim * lag] for j in range(x.shape[0])])

        # Find the false nearest neighbor percentages
        ratio = np.abs(vec1 - vec2) / r
        fnn.append(np.where(ratio > dist_tol)[0].size / x.shape[0] * 100)

    fnn = np.array(fnn)
    embed_dim = np.where(np.isclose(fnn, np.array([0])))[0][0] + 1
    if get_fnn:
        return embed_dim, fnn
    return embed_dim


def lyapunov(data, ts, lag=None, dim=None, num_states=100, get_divergence=True):
    """
    Estimate the Lyapunov exponent according to Rosenstein algorithm as follows:
    Given a time series x = [x_{0}, x_{1}, ..., x_{N - 1}],
    1-) If not provided, obtain the optimum lag and embedding dimension for the time series.
    2-) Reconstruct the attractor using method of delays, i.e. map each x_{i} in x to X_{i}
        such that X_{i} = [x_{i}, x_{i + d}, ..., x_{i + (m - 1) * d}]
        where m is the embedding dimension and d is the delay.
    3-) For each X_{i} in embedded space, find nearest neighbors X_{j}
    4-) Measure average separation <log(d_{j}(i))> of neighbors at time i*ts for i=0, ..., N - 1
        and j=0, ..., M-1. We have <log(d_{j}(i))> = sum(log(d_{j}(i))) / M,
        where d_{j}(i) = d_{j}(0) * exp(lamda * i * delta). Here d_j{0} is the initial separation
        between the jth nearest-neighbors.
    5-) Use least squares to fit a line to the <log(d_{j}(i))>. Maximum Lyapunov exponent is
        estimated to be the slope of that line.

    :param data: np.array,
        time series
    :param ts: float,
        sampling period
    :param lag: int,
        amount of lag for delay embedding.
    :param dim: int,
        embedding dimension for delay embedding.
    :param num_states: int,
        number of steps for which the the embedded orbit will advance in time.
    :param get_divergence: bool,
        if True, average nearest neighbor divergence is returned.
    :return: float,
        maximum Lyapunov exponent
    """

    assert isinstance(data, np.ndarray), \
        'Expected type for time series {}, got'.format(np.ndarray, type(data))
    assert data.ndim == 1, \
        'Expected ndim for data {}, got {}'.format(1, data.ndim)

    if lag is None:
        lag = optimum_delay(data)
    if dim is None:
        dim = embedding_dimension(data, lag)

    # Construct the embedded orbit
    orbit = delay_embedding(data, dim, lag=lag)

    # Truncate the orbit so that nearest neighbors  will evolve in time
    m = orbit.shape[0]  # number of vectors in embedded space
    num_steps = m - num_states + 1
    orbit_trunc = orbit[:num_steps]

    # Find nearest neighbors
    nn_indices = nearest_neighbors(orbit_trunc)

    # Advance the embedded orbit in time and calculate the average divergence
    divergence = np.zeros((num_states, num_steps), dtype=float)
    for i in range(num_states):
        index = np.arange(num_steps) + i
        nn_index = nn_indices + i
        divergence[i] = distance(orbit[index], orbit[nn_index])

    # Calculate the maximum Lyapunov exponent
    divergence[divergence > 0] = np.log(divergence[divergence > 0])
    y = divergence.mean(axis=1) / ts
    x = np.arange(len(y))
    lyap_exp = np.polyfit(x, y, 1)[0]  # find maximum Lyapunov exponent by least squares fit

    if get_divergence:
        return lyap_exp, y
    else:
        return lyap_exp
