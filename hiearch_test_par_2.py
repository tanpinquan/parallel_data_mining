import numpy as np
import pandas as pd
import hiearchal_seq
import hiearch_par
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

if __name__ == '__main__':

    # n = 125
    # cluster1 = np.random.randn(n, 2)
    # cluster2 = np.random.randn(n, 2) + [0, 10]
    # cluster3 = np.random.randn(n, 2) + [10, 0]
    # cluster4 = np.random.randn(n, 2) + [10, 10]
    #
    # data = np.concatenate((cluster1, cluster2, cluster3, cluster4))

    # data = np.load('testData16.npy')
    data = np.load('clusteringData500.npy')
    print('Running Parallel Hierarchical Clustering')

    data = pd.DataFrame(data)

    dist = np.ndarray((data.shape[0], data.shape[0]))
    minDist = [0]

    numProcesses = 4


    dataQueue = mp.Queue()
    resultsQueue = mp.Queue()
    start_time = time.time()

    index_split = np.array_split(np.array(range(len(data))),numProcesses)

    processes = [mp.Process(target=hiearch_par.compute_dist, args=(dataQueue, resultsQueue))
                 for i in range(numProcesses)]

    for p in processes:
        p.start()

    for i in range(numProcesses):
        # startIndex = i * dataPerProcess
        # endIndex = min((i + 1) * dataPerProcess, len(data))
        # index = list(range(startIndex, endIndex))
        dataQueue.put((index_split[i], data))
        # print(startIndex, endIndex)
        # print('putting ', i)

    for i in range(0, numProcesses):
        indexTemp, distTemp = resultsQueue.get()
        dist[indexTemp] = distTemp
        # print(i, indexTemp)

    for p in processes:
        p.join()

    dist_time = time.time() - start_time

    clusterAssignments = np.linspace(0, data.shape[0] - 1, num=data.shape[0], dtype=int)
    clusterSplit = np.array_split(clusterAssignments, numProcesses)
    uniqueClusters = clusterAssignments.copy()
    numClusters = len(uniqueClusters)

    while numClusters > 1:
        nearestNeighInd, nearestNeighDist = hiearchal_seq.get_nearest_clusters(dist, uniqueClusters)
        mergeInd, minDist = hiearchal_seq.get_merge_clusters(nearestNeighInd, nearestNeighDist, minDist)
        dist, clusterAssignments = hiearchal_seq.update_clusters(dist, mergeInd, clusterAssignments)
        uniqueClusters = np.unique(clusterAssignments[-1, :])
        numClusters = len(uniqueClusters)

    totalTime = (time.time() - start_time)
    print('dist time', dist_time)

    print('total time', totalTime)

    """Plot initial data and centroids"""
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.show()

    numClusters = 4
    selectedAssignment = clusterAssignments[-numClusters, :]
    hiearchal_seq.plot_clusters(data, selectedAssignment, 'Modified Parallel hiearch: ' + str(numClusters) + ' clusters')
