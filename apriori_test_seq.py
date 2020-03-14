import csv
from collections import defaultdict, OrderedDict
import apriori_seq
import apriori_par
import time
import multiprocessing as mp


if __name__ == '__main__':

    transactionDict = defaultdict(set)
    countList = []
    itemCount1 = {}
    minSupport = 100
    dataLength = 20636
    count = 0

    mergedCountDic = {}
    smallFileName = 'AssocRulesSmall.txt'
    fullFileName = 'Association Rules Data 1.txt'
    print('Running Sequential A Priori')

    '''Read data and initialise counts'''
    with open(fullFileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if count<dataLength:
                count += 1
                transactionDict[row[0]] = set(row[1:])
                for item in row[1:]:
                    itemCount1[frozenset([item])] = 0

    '''Sequential Part'''
    start_time = time.time()

    itemCountSeqBase = itemCount1.copy()
    countList.append(itemCountSeqBase)
    i = 0
    while len(countList[i]) != 0:
        print(i)
        countList[i] = apriori_seq.count_frequent_n_items(transactionDict, countList[i])
        countList[i] = apriori_seq.prune_count(countList[i], minSupport)
        countList.append(apriori_seq.create_count_dict(countList[i]))
        mergedCountDic.update(countList[i])
        i += 1

    seq_time = (time.time() - start_time)
    print(len(mergedCountDic), 'rules found for support of', minSupport)
    print('sequential time:', seq_time)
