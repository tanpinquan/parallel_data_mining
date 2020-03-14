import numpy as np
import matplotlib.pyplot as plt


def compute_dist_old(data_points):
    dist = np.ndarray((data_points.shape[0], data_points.shape[0]))
    nearest_neigh_ind = np.ndarray((data_points.shape[0], 1), dtype=int)
    nearest_neigh_dist = np.ndarray((data_points.shape[0], 1))
    for i, point in data_points.iterrows():
        dist[i, :] = ((data_points - point) ** 2).sum(axis=1)
        dist[i, i] = float("inf")
        ind = np.argmin(dist[i, :])
        nearest_neigh_ind[i] = ind
        nearest_neigh_dist[i] = dist[i, ind]
        # dist[i, i:] = float("inf")

    return dist, nearest_neigh_ind, nearest_neigh_dist


def compute_dist(data_points):
    dist = np.ndarray((data_points.shape[0], data_points.shape[0]))
    for i, point in data_points.iterrows():
        dist[i, :] = ((data_points - point) ** 2).sum(axis=1)
        dist[i, i] = float("inf")

    return dist


def get_nearest_clusters(dist, clusters):
    nearest_neigh_ind = np.ndarray((dist.shape[0], 1), dtype=int)
    nearest_neigh_dist = np.full((dist.shape[0], 1), np.inf)

    for i in clusters:
        ind = np.argmin(dist[i, :])
        nearest_neigh_ind[i] = ind
        nearest_neigh_dist[i] = dist[i, ind]

    return nearest_neigh_ind, nearest_neigh_dist


def get_merge_clusters(nearest_neigh_ind, nearest_neigh_dist, min_dist):
    nearest_dist = np.amin(nearest_neigh_dist)
    merge_ind = nearest_neigh_ind[nearest_neigh_dist == nearest_dist]
    min_dist.append(np.amin(nearest_neigh_dist))

    return merge_ind, min_dist


def update_clusters(dist, merge_ind, cluster_assignment):
    if len(cluster_assignment.shape) == 1:
        new_cluster_assignment = cluster_assignment.copy()
    else:
        new_cluster_assignment = cluster_assignment[-1, :].copy()

    cluster1 = new_cluster_assignment[int(merge_ind[0])]
    cluster2 = new_cluster_assignment[merge_ind[1]]
    replace_index = new_cluster_assignment == cluster1
    replace_index2 = new_cluster_assignment == cluster2
    new_cluster_assignment[replace_index] = cluster2
    dist[replace_index, cluster2] = float("inf")
    dist[replace_index2, cluster1] = float("inf")

    updated_dist = np.amin(dist[:, [cluster1, cluster2]], axis=1)
    dist[:, cluster2] = updated_dist
    dist[cluster2, :] = updated_dist
    dist[:, cluster1] = float("inf")
    dist[cluster1, :] = float("inf")

    cluster_assignment = np.vstack((cluster_assignment, new_cluster_assignment))

    return dist, cluster_assignment


def update_clusters_old(dist, min_dist, cluster_assignment):
    min_dist.append(np.amin(dist))
    (min_row_loc, min_col_loc) = np.where(dist == min_dist[-1])

    if len(cluster_assignment.shape) == 1:
        new_cluster_assignment = cluster_assignment.copy()
    else:
        new_cluster_assignment = cluster_assignment[-1, :].copy()

    for row, col in zip(min_row_loc, min_col_loc):
        cluster1 = new_cluster_assignment[row]
        cluster2 = new_cluster_assignment[col]
        replace_index = new_cluster_assignment == cluster1
        replace_index2 = new_cluster_assignment == cluster2
        new_cluster_assignment[replace_index] = cluster2
        dist[replace_index, cluster2] = float("inf")
        dist[replace_index2, cluster1] = float("inf")

        dist[:, cluster2] = np.amin(dist[:, [cluster1, cluster2]], axis=1)
        dist[:, cluster1] = float("inf")

    cluster_assignment = np.vstack((cluster_assignment, new_cluster_assignment))

    return dist, min_dist, cluster_assignment


def plot_clusters(data, centroid_assignments, title):
    centroid_labels = np.unique(centroid_assignments)
    for centroid in centroid_labels:
        selected_points = data[centroid_assignments == centroid]
        plt.scatter(selected_points.iloc[:, 0], selected_points.iloc[:, 1])

    plt.title(title)
    plt.show()
