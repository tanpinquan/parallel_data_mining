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

    data = np.load('clusteringData2000.npy')
    data = pd.DataFrame(data)
    print('Running Parallel Hierarchical Clustering')

    dist = np.ndarray((data.shape[0], data.shape[0]))
    minDist = [0]

    numProcesses = 2
    dataPerProcess = round(len(data) / numProcesses)
    dataQueue = mp.Queue()
    resultsQueue = mp.Queue()

    start_par = time.time()

    processes = [mp.Process(target=hiearch_par.compute_dist, args=(dataQueue, resultsQueue))
                 for i in range(numProcesses)]

    for p in processes:
        p.start()

    for i in range(numProcesses):
        startIndex = i * dataPerProcess
        endIndex = (i + 1) * dataPerProcess
        index = list(range(startIndex, endIndex))
        dataQueue.put((index, data))
        # print('putting ', i)

    for i in range(0, numProcesses):
        indexTemp, distTemp = resultsQueue.get()
        dist[indexTemp] = distTemp
        # print(i, indexTemp)

    for p in processes:
        p.join()

    ''''''
    clusterAssignments = np.linspace(0, data.shape[0] - 1, num=data.shape[0], dtype=int)
    clusterSplit = np.array_split(clusterAssignments, numProcesses)
    uniqueClusters = clusterAssignments.copy()
    numClusters = len(uniqueClusters)

    nearestNeighInd = np.ndarray((data.shape[0], 1), dtype=int)
    nearestNeighDist = np.full((data.shape[0], 1), np.inf)

    processes = [mp.Process(target=hiearch_par.update_dist_and_get_nearest_clusters, args=(dataQueue, resultsQueue))
                 for i in range(numProcesses)]

    for p in processes:
        p.start()

    for i in range(numProcesses):
        dataQueue.put((dist, clusterAssignments,clusterSplit[i]))
        # print('putting ', i)

    for i in range(0, numProcesses):
        clusterInd, nearestNeighIndTemp, nearestNeighDistTemp = resultsQueue.get()
        nearestNeighInd[clusterInd] = nearestNeighIndTemp
        nearestNeighDist[clusterInd] = nearestNeighDistTemp


    mergeInd, minDist = hiearchal_seq.get_merge_clusters(nearestNeighInd, nearestNeighDist, minDist)
    dist, clusterAssignments = hiearchal_seq.update_clusters(dist, mergeInd, clusterAssignments)
    uniqueClusters = np.unique(clusterAssignments[-1, :])
    numClusters = len(uniqueClusters)
    clusterSplit = np.array_split(uniqueClusters, numProcesses)

    while numClusters > 1:
        nearestNeighInd = np.ndarray((data.shape[0], 1), dtype=int)
        nearestNeighDist = np.full((data.shape[0], 1), np.inf)
        for i in range(numProcesses):
            dataQueue.put((mergeInd, clusterSplit[i]))
            # print('putting ', i)

        for i in range(0, numProcesses):
            clusterInd, nearestNeighIndTemp, nearestNeighDistTemp = resultsQueue.get()
            nearestNeighInd[clusterInd] = nearestNeighIndTemp
            nearestNeighDist[clusterInd] = nearestNeighDistTemp

        # print(nearestNeighInd)
        # print(nearestNeighDist)
        mergeInd, minDist = hiearchal_seq.get_merge_clusters(nearestNeighInd, nearestNeighDist, minDist)
        dist, clusterAssignments = hiearchal_seq.update_clusters(dist, mergeInd, clusterAssignments)
        uniqueClusters = np.unique(clusterAssignments[-1, :])
        numClusters = len(uniqueClusters)
        clusterSplit = np.array_split(uniqueClusters, numProcesses)

        # print('main', numClusters)
    for p in processes:
        p.join()




    totalTime = (time.time() - start_par)
    print('time', totalTime)

    """Plot initial data and centroids"""
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.show()

    numClusters = 4
    selectedAssignment = clusterAssignments[-numClusters, :]
    hiearchal_seq.plot_clusters(data, selectedAssignment, 'Parallel hiearch: ' + str(numClusters) + ' clusters')
