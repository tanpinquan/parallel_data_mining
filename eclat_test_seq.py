import csv
from collections import defaultdict, OrderedDict
import eclat_seq
import eclat_par
import time
import multiprocessing as mp

minSupport = 100
dic = defaultdict(list)
dic1 = {}
dic2 = {}
dic2a = {}

dicList = []
mergedDic = {}
dataLength = 20636
count = 0
print('Running Sequential ECLAT')

with open('Association Rules Data 1.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        if count < dataLength:
            [dic[item].append(row[0]) for item in row[1:]]
            count+=1

for key, value in dic.items():
    dic1[frozenset([key])] = set(value)

'''Get 1-frequent itemsets'''
dic1 = eclat_seq.prune_dict(dic1, minSupport)
dicList.append(dic1)
mergedDic = dic1.copy()

'''Sequential part'''
start_time = time.time()
mergedDic, dicList = eclat_seq.get_frequent_itemsets(dic1, minSupport)


elapsed_part_1 = (time.time() - start_time)


print(len(mergedDic), 'rules found for support of ', str(minSupport))
print('Sequential time', elapsed_part_1)

