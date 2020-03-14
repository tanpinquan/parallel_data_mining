import csv
from collections import defaultdict, OrderedDict
import apriori_seq
import apriori_par
import time
import multiprocessing as mp


if __name__ == '__main__':

    transactionDict = defaultdict(set)
    countList = []
    # itemCount = {}
    # itemCount2 = {}
    # itemCount3 = {}
    itemCount1 = {}
    minSupport = 100
    dataLength = 20636
    count = 0
    numProcesses = 4

    mergedCountDic = {}
    smallFileName = 'AssocRulesSmall.txt'
    fullFileName = 'Association Rules Data 1.txt'
    
    print('Running Parallel Apriori:', numProcesses, 'processes')
    '''Read data and initialise counts'''
    with open(fullFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if count<dataLength:
                count +=1
                transactionDict[row[0]] = set(row[1:])
                for item in row[1:]:
                    itemCount1[frozenset([item])] = 0


    ''' Parallel Part'''
    start_time = time.time()

    itemCountParBase = itemCount1.copy()
    countListPar = [itemCountParBase.copy()]
    countListProc = [countListPar[-1].copy() for i in range(numProcesses)]
    mergedCountDicPar = {}

    transactionDictPartitions = apriori_par.partition_sets(transactionDict, numProcesses)

    dataQueue = mp.Queue()
    resultsQueue = mp.Queue()

    # for iteration in range(2):
    while len(countListPar[-1]) != 0:

        processes = [mp.Process(target=apriori_par.parallel_apriori, args=(dataQueue, resultsQueue))
                     for i in range(numProcesses)]

        for p in processes:
            p.start()

        for i in range(numProcesses):
            dataQueue.put((transactionDictPartitions[i], countListProc[i]))

        for i in range(0, numProcesses):
            countListProc[i].update(resultsQueue.get())

        for p in processes:
            p.join()

        # start_overhead_time = time.time()

        countListPar[-1] = apriori_par.merge_proc_counts(countListProc)
        countListPar[-1] = apriori_seq.prune_count(countListPar[-1], minSupport)
        mergedCountDicPar.update(countListPar[-1])

        countListPar.append(apriori_seq.create_count_dict(countListPar[-1]))
        countListProc = [countListPar[-1].copy() for i in range(numProcesses)]

        # overhead_time = (time.time() - start_overhead_time)
        # print('overhead time', overhead_time)

    par_time = (time.time() - start_time)

    print(len(mergedCountDicPar), 'rules found for support of', minSupport)

    print('parallel time', par_time)
