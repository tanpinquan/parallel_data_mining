import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import kmeans_seq


# class KMeansData:
#     def __init__(self, data, centroids):
#         self.data = data
#         self.centroids = centroids
#
#
# class CentroidAssignments:
#     def __init__(self, index, assignment):
#         self.index = index
#         self.assignment = assignment


def worker(num):
    """worker function"""
    print('Start:', num, 'no wait')
    print('End:', num, 'no wait')

    return


def worker_wait(num):
    """worker function"""
    print('Start:', num, 'waiting for 2sec')
    time.sleep(2)
    print('End:', num, '2 sec over')
    return


def assign_centroids(data_q, results_q):
    """worker function"""
    process_name = mp.current_process().name
    # print('Start:', process_name)
    data, centroids = data_q.get()
    # print('Data:', data.head())
    # print('Centroids:', centroids)

    dist = kmeans_seq.compute_distance(data, centroids)
    data_index = list(data.index.values)

    assignment = dist.argmin(axis=1)
    # print('Index1: ', type(data_index), data_index)
    # print('Assignment: ', assignment)

    results_q.put((data_index, assignment))

    # print('End:', process_name)
    return


def assign_centroids_2(data_q, centroids_q, results_q):
    """worker function"""
    process_name = mp.current_process().name
    # print('Start:', process_name)
    data, num_iter = data_q.get()
    # print('Data:', data.head())
    # print('Centroids:', centroids)
    for i in range(num_iter):
        centroids = centroids_q.get()
        # print('centroid i', (centroids,i))
        dist = kmeans_seq.compute_distance(data, centroids)
        data_index = list(data.index.values)

        assignment = dist.argmin(axis=1)
        # print('Index1: ', type(data_index), data_index)
        # print('Assignment: ', assignment)

        results_q.put((data_index, assignment))

    # print('End:', process_name)
    return


# def assign_centroids_3(data_q, results_q):
#     """worker function"""
#     process_name = mp.current_process().name
#
#     data, centroids = data_q.get()
#
#     dist = kmeans_seq.compute_distance(data, centroids)
#     data_index = list(data.index.values)
#
#     assignment = dist.argmin(axis=1)
#
#     results_q.put((data_index, assignment))
#
#     return


def assign_centroids_4(data_q, centroids_q, results_q):
    """worker function"""
    process_name = mp.current_process().name
    # print('Start:', process_name)
    data, num_iter = data_q.get()
    # print('Data:', data.head())
    # print('Centroids:', centroids)
    last_iter = False
    while not last_iter :
        last_iter, centroids = centroids_q.get()
        # print('centroid i', (centroids,i))
        dist = kmeans_seq.compute_distance(data, centroids)
        data_index = list(data.index.values)

        assignment = dist.argmin(axis=1)
        # print('Index1: ', type(data_index), data_index)
        # print('Assignment: ', assignment)

        results_q.put((data_index, assignment))

    # print('End:', process_name)
    return


def update_centroids(data, data_index, centroid_assignments, num_centroids, num_features):
    centroid_locations = np.ndarray((num_centroids, num_features))
    for centroid in range(0, num_centroids):
        selected_index = data_index[centroid_assignments == centroid]
        selected_points = data.iloc[selected_index]
        # print(selected_points.shape)
        centroid_locations[centroid, :] = selected_points.mean(axis=0)

    return centroid_locations


def plot_clusters(data, centroid_assignments, data_index, centroids, title):
    centroid_labels = np.unique(centroid_assignments)
    for centroid in centroid_labels:
        selected_points = data.loc[data_index[centroid_assignments == centroid]]
        plt.scatter(selected_points.iloc[:, 0], selected_points.iloc[:, 1])

    # ax = data.loc[data_index[centroid_assignments == 0]].plot.scatter(x=0, y=1, c='r')
    # data.loc[data_index[centroid_assignments == 1]].plot.scatter(x=0, y=1, c='g', ax=ax)
    # data.loc[data_index[centroid_assignments == 2]].plot.scatter(x=0, y=1, c='b', ax=ax)
    # data.loc[data_index[centroid_assignments == 3]].plot.scatter(x=0, y=1, c='m', ax=ax)

    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], c='k', marker='x', s=100)
    plt.title(title)
    plt.show()


def assign_centroids_pool(data, centroids):
    # dist = np.ndarray(centroids.shape[0])
    dist = np.ndarray(len(centroids))

    # print('data', data)
    # print('centroids',centroids)
    for i, centroid in enumerate(centroids):
        # print(data)
        # print('dist',sum((data - centroid) ** 2))
        dist[i] = sum((data - centroid) ** 2)

    return dist.argmin()


def update_centroids_pool(centroid_id, data, centroid_assignments):
    selected_points = data[centroid_assignments == centroid_id]
    centroid = selected_points.mean(axis=0)

    return centroid
