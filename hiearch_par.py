import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hiearchal_seq
import multiprocessing as mp


def compute_dist(data_q, results_q):
    """worker function"""
    index, data = data_q.get()
    dist = np.ndarray((len(index), data.shape[0]))

    for i, data_ind in enumerate(index):
        dist[i, :] = ((data - data.iloc[data_ind, :]) ** 2).sum(axis=1)
        dist[i, data_ind] = float("inf")

    results_q.put((index, dist))

    return


def get_nearest_clusters(data_q, results_q):
    dist, clusters = data_q.get()

    nearest_neigh_ind = np.ndarray((clusters.shape[0], 1), dtype=int)
    nearest_neigh_dist = np.full((clusters.shape[0], 1), np.inf)

    for i, cluster in enumerate(clusters):
        ind = np.argmin(dist[cluster, :])
        nearest_neigh_ind[i] = ind
        nearest_neigh_dist[i] = dist[cluster, ind]

    results_q.put((clusters, nearest_neigh_ind, nearest_neigh_dist))

    return


def update_dist_and_get_nearest_clusters(data_q, results_q):
    dist, cluster_assignments, clusters = data_q.get()

    nearest_neigh_ind = np.ndarray((clusters.shape[0], 1), dtype=int)
    nearest_neigh_dist = np.full((clusters.shape[0], 1), np.inf)

    for i, cluster in enumerate(clusters):
        ind = np.argmin(dist[cluster, :])
        nearest_neigh_ind[i] = ind
        nearest_neigh_dist[i] = dist[cluster, ind]

    results_q.put((clusters, nearest_neigh_ind, nearest_neigh_dist))

    num_clusters = len(cluster_assignments)

    while num_clusters > 2:
        merge_ind, clusters = data_q.get()
        dist, cluster_assignments = hiearchal_seq.update_clusters(dist, merge_ind, cluster_assignments)
        # print(clusters)
        # print(merge_ind)
        # print(cluster_assignments)
        # print(dist)

        nearest_neigh_ind = np.ndarray((clusters.shape[0], 1), dtype=int)
        nearest_neigh_dist = np.full((clusters.shape[0], 1), np.inf)

        for i, cluster in enumerate(clusters):
            ind = np.argmin(dist[cluster, :])
            nearest_neigh_ind[i] = ind
            nearest_neigh_dist[i] = dist[cluster, ind]

        results_q.put((clusters, nearest_neigh_ind, nearest_neigh_dist))


        # print('min dist:', nearest_neigh_dist)
        # print('min ind:', nearest_neigh_ind)
        unique_clusters = np.unique(cluster_assignments[-1, :])
        num_clusters = len(unique_clusters)
        # print('proc', num_clusters)

    return
