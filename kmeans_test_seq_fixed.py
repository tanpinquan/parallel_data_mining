import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import kmeans_par
import kmeans_seq
from itertools import repeat

# print("Number of processors: ", mp.cpu_count())
# generate test data


numIter = 10
# n = 2000
# cluster1 = np.random.randn(n, 2) + [0, 0]
# cluster2 = np.random.randn(n, 2) + [0, 7.5]
# cluster3 = np.random.randn(n, 2) + [7.5, 0]
# cluster4 = np.random.randn(n, 2) + [7.5, 7.5]
#
# data = np.concatenate((cluster1, cluster2, cluster3, cluster4))


data = np.load('clusteringData2000.npy')
data = pd.DataFrame(data)

numFeatures = data.shape[1]
numPoints = data.shape[0]

"""Initialize centroids randomly"""
numCentroids = 4
centroids = np.array(data.sample(n=numCentroids))
centroids_seq = centroids.copy()
centroids_pool = centroids.copy()

"""Plot initial data and centroids"""
data.plot.scatter(x=0, y=1)
plt.scatter(x=[0, 0, 7.5, 7.5], y=[0, 7.5, 0, 7.5], c='r', marker='+', s=100)
# plt.scatter(x=centroids[:, 0], y=centroids[:, 1], c='r', marker='+', s=100)
plt.show()
dist = np.ndarray((data.shape[0], numCentroids))

start_par = time.time()
print('Running Sequential K Means')

""""Sequential implementation"""
start_seq = time.time()
for iteration in range(numIter):
    start_time = time.time()

    centroid_assignments_seq = kmeans_seq.assign_centroid(data, centroids_seq)
    elapsed_part1 = (time.time() - start_time)

    # kmeans_seq.plot_clusters(data, centroid_assignments_seq, centroids_seq, title='Seq '+str(iteration))

    start_time = time.time()
    centroids_seq = kmeans_seq.update_centroids(data, centroid_assignments_seq, numCentroids, numFeatures)
    elapsed_part2 = (time.time() - start_time)

elapsed_seq = (time.time() - start_seq)

kmeans_seq.plot_clusters(data, centroid_assignments_seq, centroids_seq, title='K Means Seq')

print('--------------------')
# print('pool time:', elapsed_pool)
print('sequential time:', elapsed_seq)
