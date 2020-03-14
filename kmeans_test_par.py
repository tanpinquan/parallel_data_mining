import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import kmeans_par
import kmeans_seq

# print("Number of processors: ", mp.cpu_count())
# generate test data


if __name__ == '__main__':
    numIter = 0
    # n = 1000
    # cluster1 = np.random.randn(n, 2)
    # cluster2 = np.random.randn(n, 2) + [0, 10]
    # cluster3 = np.random.randn(n, 2) + [10, 0]
    # cluster4 = np.random.randn(n, 2) + [10, 10]
    #
    # data = np.concatenate((cluster1, cluster2, cluster3, cluster4))
    # data_pool = np.concatenate((cluster1, cluster2, cluster3, cluster4))
    data = np.load('clusteringData2000.npy')

    data = pd.DataFrame(data)

    numFeatures = data.shape[1]
    numPoints = data.shape[0]

    """Initialize centroids randomly"""
    numCentroids = 4
    centroids = np.array(data.sample(n=numCentroids))

    """Plot initial data and centroids"""
    # data.plot.scatter(x=0, y=1)
    # plt.scatter(x=centroids[:, 0], y=centroids[:, 1], c='r', marker='+', s=100)
    # plt.show()


    dist = np.ndarray((data.shape[0], numCentroids))
    numProcesses = 4
    GlobalError =100.0
    lastIter = False
    stopIter = False
    print('--------------------\nProcess Implementation: ', numProcesses, 'processes')

    dataQueue = mp.Queue()
    centroidsQueue = mp.Queue()
    resultsQueue = mp.Queue()
    dataPerProcess = round(len(data) / numProcesses)


    start_par = time.time()


    processes = [mp.Process(target=kmeans_par.assign_centroids_4, args=(dataQueue, centroidsQueue, resultsQueue))
                 for i in range(numProcesses)]

    for p in processes:
        p.start()

    for i in range(numProcesses):
        startIndex = i * dataPerProcess
        endIndex = (i + 1) * dataPerProcess
        dataQueue.put((data[startIndex:endIndex], numIter))

    while not stopIter:
        if lastIter:
            stopIter = True
        dataIndex = []
        centroid_assignments = []
        temp_centroids=centroids.copy()

        for i in range(0, numProcesses):
            centroidsQueue.put((lastIter, centroids))

        for i in range(0, numProcesses):
            indexTemp, assignmentTemp = resultsQueue.get()
            dataIndex = np.concatenate((dataIndex, indexTemp))
            centroid_assignments = np.concatenate((centroid_assignments, assignmentTemp))

        centroids = kmeans_par.update_centroids(data, dataIndex, centroid_assignments, numCentroids, numFeatures)
        GlobalError=kmeans_seq.update_global_error(temp_centroids, centroids)
        if GlobalError < 0.0001:
            lastIter = True
        numIter=numIter+1
        print('iter:', numIter)
        # ax = data.loc[dataIndex[centroid_assignments == 0]].plot.scatter(x=0, y=1, c='r')
        # data.loc[dataIndex[centroid_assignments == 1]].plot.scatter(x=0, y=1, c='g', ax=ax)
        # data.loc[dataIndex[centroid_assignments == 2]].plot.scatter(x=0, y=1, c='b', ax=ax)
        # data.loc[dataIndex[centroid_assignments == 3]].plot.scatter(x=0, y=1, c='m', ax=ax)
        #
        # plt.scatter(x=centroids[:, 0], y=centroids[:, 1], c='k', marker='x', s=100)
        # plt.title('Parallel ' + str(iteration))
        # plt.show()
    print('before join')
    for p in processes:
        p.join()
    print('after join')
    elapsed_par = (time.time() - start_par)

    print('--------------------')
    print('par time:', elapsed_par)
    # print('pool time:', elapsed_pool)

    ''' Plot result'''
    kmeans_par.plot_clusters(data, centroid_assignments, dataIndex, centroids, 'K Means Par')
