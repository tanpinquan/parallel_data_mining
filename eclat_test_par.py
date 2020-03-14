import csv
from collections import defaultdict, OrderedDict
import eclat_seq
import eclat_par
import time
import multiprocessing as mp

if __name__ == '__main__':
    minSupport = 10
    dic = defaultdict(list)
    dic1 = {}
    dic2 = {}
    dic2a = {}

    dicList = []
    mergedDic = {}
    dataLength = 20636
    count = 0
    numProcesses = 4
    print('Running Parallel ECLAT')

    with open('Association Rules Data 1.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if count < dataLength:
                [dic[item].append(row[0]) for item in row[1:]]
                count += 1

    for key, value in dic.items():
        dic1[frozenset([key])] = set(value)

    '''Get 1-frequent itemsets'''
    dic1 = eclat_seq.prune_dict(dic1, minSupport)
    dicList.append(dic1)
    mergedDic = dic1.copy()

    '''Sequential part'''
    start_time = time.time()

    '''Parallel part'''
    '''Get 2-frequent itemsets'''
    start_time = time.time()

    dic2a = eclat_seq.merge_tids_2(dic1, minSupport)
    elapsed_part_2 = (time.time() - start_time)

    start_time = time.time()

    dic2, num_sets = eclat_par.generate_two_itemsets(dic1, minSupport)
    dic_partitions = eclat_par.partition_sets(dic2, num_sets, numProcesses)
    elapsed_part_3 = (time.time() - start_time)
    start_time = time.time()

    dicParList = []
    mergedDicPar = dic1.copy()
    dataQueue = mp.Queue()
    resultsQueue = mp.Queue()
    processes = [mp.Process(target=eclat_par.parallel_eclat, args=(dataQueue, resultsQueue))
                 for i in range(numProcesses)]

    for p in processes:
        p.start()

    for i in range(numProcesses):
        dataQueue.put((dic_partitions[i], minSupport))

    for i in range(0, numProcesses):
        dicParList.append(resultsQueue.get())

    for p in processes:
        p.join()

    for dic_proc in dicParList:
        mergedDicPar.update(dic_proc)

    elapsed_part_4 = (time.time() - start_time)

    # print('1-item :', str(len(dic1)), 'rules')
    # for p, dic_proc in enumerate(dicParList):
    #     print('proc', str(p), ':', len(dic_proc), 'rules')

    print(len(mergedDicPar), 'rules found for support of ', str(minSupport))

    print('parallel time', elapsed_part_2 + elapsed_part_3 + elapsed_part_4)
