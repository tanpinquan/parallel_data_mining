import numpy as np
import pandas as pd
import hiearchal_seq
import matplotlib.pyplot as plt
import time

# n = 125
# cluster1 = np.random.randn(n, 2) + [0, 0]
# cluster2 = np.random.randn(n, 2) + [0, 7.5]
# cluster3 = np.random.randn(n, 2) + [7.5, 0]
# cluster4 = np.random.randn(n, 2) + [7.5, 7.5]
#
# data = np.concatenate((cluster1, cluster2, cluster3, cluster4))

# data = np.load('testData16.npy')
data = np.load('clusteringData500.npy')
data = pd.DataFrame(data)
print('Running Sequential Hierarchical Clustering')

clusterAssignments = np.linspace(0, data.shape[0] - 1, num=data.shape[0], dtype=int)

minDist = [0]
start_time = time.time()

# dist, nearestNeighInd, nearestNeighDist = hiearchal_seq.compute_dist(data)
dist = hiearchal_seq.compute_dist(data)
dist_time = time.time() - start_time

uniqueClusters = clusterAssignments.copy()
numClusters = len(uniqueClusters)



''' Iteration 2'''
while numClusters > 1:
    nearestNeighInd, nearestNeighDist = hiearchal_seq.get_nearest_clusters(dist, uniqueClusters)
    mergeInd, minDist = hiearchal_seq.get_merge_clusters(nearestNeighInd, nearestNeighDist, minDist)
    dist, clusterAssignments = hiearchal_seq.update_clusters(dist, mergeInd, clusterAssignments)
    uniqueClusters = np.unique(clusterAssignments[-1, :])
    numClusters = len(uniqueClusters)

elapsed_seq = (time.time() - start_time)
print('dist init time', dist_time)

print('total time', elapsed_seq)
"""Plot initial data and centroids"""
plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
plt.show()

numClusters = 4
selectedAssignment = clusterAssignments[-numClusters, :]
hiearchal_seq.plot_clusters(data, selectedAssignment, 'Hierarchical Clustering: ' + str(numClusters) + ' clusters')

# numClusters = len(np.unique(clusterAssignments))
# while numClusters > 1:
#     (dist, minDist, clusterAssignments) = hiearchal_seq.update_clusters(dist, minDist, clusterAssignments)
#     numClusters = len(np.unique(clusterAssignments[-1, :]))
#     # print(numClusters)
#
#
# elapsed_seq = (time.time() - start_time)
# print('time', elapsed_seq)
#
# """Plot initial data and centroids"""
# plt.scatter(data.iloc[:,0], data.iloc[:,1])
# plt.show()
#
# numClusters = 4
# selectedAssignment = clusterAssignments[-numClusters,:]
#
# hiearchal_seq.plot_clusters(data,selectedAssignment,str(numClusters) +' clusters')
