import numpy as np
from numpy.core._multiarray_umath import ndarray
import matplotlib.pyplot as plt


def compute_distance(data, centroids):
    dist = np.ndarray((data.shape[0], centroids.shape[0]))
    # for index in data.index:
    for index in range(0, len(data)):
        for i, centroid in enumerate(centroids):
            dist[index, i] = sum((data.iloc[index, :] - centroid) ** 2)

    # dist_point = np.ndarray((centroids.shape[0]))
    # for i, centroid in enumerate(centroids):
    #     dist_point[i] = sum((row-centroid)**2)

    return dist


def assign_centroid(data, centroids):
    dist = compute_distance(data, centroids)
    return dist.argmin(axis=1)


def update_centroids(data, centroid_assignments, num_centroids, num_features):
    centroid_locations = np.ndarray((num_centroids, num_features))
    for centroid in range(0, num_centroids):
        selected_points = data.iloc[centroid_assignments == centroid]
        # print(selected_points.shape)
        centroid_locations[centroid, :] = selected_points.mean(axis=0)

    return centroid_locations


def compute_distance_pool(data, centroids):
    dist = np.ndarray((data.shape[0], centroids.shape[0]))
    # for index in data.index:
    for index in range(0, len(data)):
        for i, centroid in enumerate(centroids):
            dist[index, i] = sum((data[index, :] - centroid) ** 2)

    # dist_point = np.ndarray((centroids.shape[0]))
    # for i, centroid in enumerate(centroids):
    #     dist_point[i] = sum((row-centroid)**2)

    return dist


def plot_clusters(data, centroid_assignments, centroids, title):
    centroid_labels = np.unique(centroid_assignments)
    for centroid in centroid_labels:
        selected_points = data[centroid_assignments == centroid]
        plt.scatter(selected_points.iloc[:, 0], selected_points.iloc[:, 1])

    plt.scatter(x=np.array(centroids)[:, 0], y=np.array(centroids)[:, 1], c='k', marker='x', s=100)
    plt.title(title)

    plt.show()


def update_global_error(old_centroids, new_centroids):
    error=0
    old=np.sort(old_centroids, axis=0)
    new=np.sort(new_centroids, axis=0)
    
    error =sum((old - new) ** 2)

    return np.sum(error)
