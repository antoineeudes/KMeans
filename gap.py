import numpy as np
import matplotlib.pyplot as plt
from kmeans import *
from sklearn.datasets import make_blobs


def bb(X):
    nb_samples, nb_features = X.shape 
    bb_min = [np.min(X[:, k]) for k in range(nb_features)]
    bb_max = [np.max(X[:, k]) for k in range(nb_features)]
    return bb_min, bb_max

def compute_log_inertia(X, n_clusters, T, bb_min, bb_max,
                        random_state=0):
    """Compute the log inertia of X and X_t.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_clusters: int
        The desired number of clusters.

    T: int
        Number of draws of X_t.

    bb_min: array, shape (n_features,)
        Inferior corner of the bounding box of X.

    bb_max: array, shape (n_features,)
        Superior corner of the bounding box of X.

    random_state: int, defaults to 0.
        A random number generator instance.

    Returns
    -------
    log_inertia: float
        Log of the inertia of the K-means applied to X.

    mean_log_inertia_rand: float
        Mean of the log of the inertia of the K-means applied to the different
        X_t.

    std_log_inertia_rand: float
        Standard deviation of the log of the inertia of the K-means applied to
        the different X_t.
    """
    nb_experiences = 100

    log_inertia = np.log(kmeans(X, n_clusters, show=False)[2])
    experiences = []
    np.random.seed(random_state)
    for _ in range(nb_experiences):
        Xt = np.random.uniform(bb_min, bb_max, size=(T, 2))
        experiences.append(np.log(kmeans(Xt, n_clusters, show=False)[2]))
    mean_log_inertia_rand = np.mean(experiences)
    std_log_inertia_rand = np.std(experiences)

    return log_inertia, mean_log_inertia_rand, std_log_inertia_rand


def compute_gap(X, n_clusters_max, T=10, random_state=0):
    """Compute values of Gap and delta.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_cluster_max: int
        Maximum number of cluster to test.

    T: int, defaults 10.
        Number of draws of X_t.

    random_state: int, defaults to 0.
        A random number generator instance.

    Returns
    -------
    n_clusters_range: array-like, shape (n_clusters_max-1,)
        Array of number of clusters tested.

    gap: array-like, shape (n_clusters_max-1,)
        Return the gap values.

    delta: array-like, shape (n_clusters_max-1,)
        Return the delta values.
    """
    bb_min, bb_max = bb(X)
    log_inertias = np.zeros(n_clusters_max)
    mean_logs = np.zeros(n_clusters_max)
    std_logs = np.zeros(n_clusters_max)
    for k in range(1, n_clusters_max):
        log_inertias[k], mean_logs[k], std_logs[k] =  compute_log_inertia(X, k, T, bb_min, bb_max)
    gap = mean_logs - log_inertias
    sigma = np.sqrt((T+1)/T) * std_logs
    delta = gap - np.roll(gap-sigma, -1, axis=0)
    return list(range(n_clusters_max)), gap, delta[:-1]



def plot_result(n_clusters_range, gap, delta):
    """Plot the values of Gap and delta.

    Parameters
    ----------
    n_clusters_range: array-like, shape (n_clusters_max-1,)
        Array of number of clusters tested.

    gap: array-like, shape (n_clusters_max-1,)
        Return the gap values.

    delta: array-like, shape (n_clusters_max-1,)
        Return the delta values.
    """
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=.05, right=.98, bottom=.08, top=.98, wspace=.15,
                        hspace=.03)

    plt.subplot(121)
    plt.plot(n_clusters_range, gap)
    plt.ylabel(r'$Gap(k)$', fontsize=18)
    plt.xlabel("Number of clusters")

    plt.subplot(122)
    for x, y in zip(n_clusters_range, delta):
        plt.bar(x - .45, y, width=0.9)
    plt.ylabel(r'$\delta(k)$', fontsize=18)
    plt.xlabel("Number of clusters")

    plt.draw()


def optimal_n_clusters_search(X, n_clusters_max, T=10, random_state=0):
    """Compute the optimal number of clusters.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_cluster_max: int
        Maximum number of cluster to test.

    T: int, defaults 10.
        Number of draws of X_t.

    random_state: int, defaults to 0.
        A random number generator instance.

    Returns
    -------
    n_clusters_optimal: int
        Optimal number of clusters.
    """
    pass


if __name__ == '__main__':
    # Parameters
    random_state = 0
    n_samples, n_clusters_max = 1000, 10

    X, y = make_blobs(n_samples=n_samples, random_state=random_state,
                      centers=3)
    bb_min, bb_max = bb(X)
    print(compute_log_inertia(X, n_clusters_max, 10, bb_min, bb_max))
    n_clusters_range, gap, delta = compute_gap(X, 10)
    plot_result(n_clusters_range, gap, delta)
    plt.show()
    

    