import numpy as np
from matplotlib import pyplot as plt
import time


def compute_labels(X, centroids):
    """Compute labels.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    centroids: array-like, shape (n_clusters, n_features)
        The estimated centroids.

    Returns
    -------
    labels : array, shape (n_samples,)
        The labels of each sample
    """
    # Q1: Implement K-means
    labels = np.zeros(len(X))
    for it_x in range(len(X)):
        dists = np.linalg.norm(centroids-X[it_x], axis=1)
        labels[it_x] = np.argmin(dists)
    return labels


def compute_inertia_centroids(X, labels):
    """Compute inertia and centroids.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like, shape (n_saples,)
        The labels of each sample.

    Returns
    -------
    inertia: float
        The inertia.

    centroids: array-like, shape (n_clusters, n_features)
        The estimated centroids.
    """
    # Q1: Implement K-means
    nb_clusters = int(np.max(labels)+1)
    inertia = 0
    _, nb_features = X.shape
    barycenters = np.zeros((nb_clusters, nb_features))

    for k in range(nb_clusters):
        points_in_cluster = X[labels==k] 
        barycenters[k] = np.sum(points_in_cluster, axis=0)/len(points_in_cluster)
        inertia += np.sum(np.linalg.norm(points_in_cluster-barycenters[k], axis=1)**2)
    return inertia, barycenters


def kmeans(X, n_clusters, max_iter=100, tol=1e-7, random_state=42, show=False):
    """Estimate position of centroids and labels.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.

    n_clusters: int
        The desired number of clusters.

    max_iter: int, defaults 100.
        Max number of update.

    tol: float, defaults 1e-7.
        The tolerance to check convergence.

    random_state: int, defaults to 42.
        A random number generator instance.

    Returns
    -------
    centroids: array-like, shape (n_clusters, n_features)
        The estimated centroids.

    labels: array-like, shape (n_samples,)
        The estimated labels.

    inertia: float
        The inertia.
    """
    # Q1: Implement K-means
    nb_samples, nb_features = X.shape
    np.random.seed(random_state)
    initial_centroids_indexes = np.random.choice(nb_samples, n_clusters)
    centroids = X[initial_centroids_indexes]
    inertia = float('inf')
    inertias = []

    for iteration in range(max_iter):
        labels = compute_labels(X, centroids)
        new_inertia, centroids = compute_inertia_centroids(X, labels)
        inertias.append(new_inertia)

        if abs(new_inertia - inertia) < tol:
            break

        inertia = new_inertia
    if show:
        plt.plot(range(len(inertias)), inertias)
        plt.show()
    return centroids, labels, new_inertia

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn import cluster

    # Parameters
    random_state = 0
    n_samples = 1000
    color = 'rgbcmyk'

    # Generate data
    X, y = make_blobs(n_samples=n_samples, random_state=random_state,
                      centers=3)

    # Q1-Q4 Apply K-means to X
    time1 = time.clock()
    _, labels, _ = kmeans(X, 7, show=False)
    time2 = time.clock()
    print("A la main : {}".format(time2-time1))
    time3 = time.clock()
    kmeans = cluster.KMeans(n_clusters=7, n_init=10)
    kmeans.fit(X)
    time4 = time.clock()
    print("Sklearn : {}".format(time4-time3))
    for i in range(int(np.max(labels)+1)):
        X_lab = X[labels==i]
        plt.subplot(121)
        plt.scatter(X_lab[:,0], X_lab[:,1], color=color[i])

    sklabels = kmeans.labels_
    for i in range(int(np.max(sklabels)+1)):
        X_lab = X[sklabels==i]
        plt.subplot(122)
        plt.scatter(X_lab[:,0], X_lab[:,1], color=color[i])
    plt.show()