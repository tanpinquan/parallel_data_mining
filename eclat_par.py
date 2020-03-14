from collections import defaultdict
import eclat_seq


def generate_two_itemsets(dic, min_support):
    dic_list = []
    dic_keys = list(dic.keys())
    num_sets = 0;
    for i in range(len(dic_keys)):
        out_dic = {}

        for j in range(i + 1, len(dic_keys)):
            key1 = dic_keys[i]
            key2 = dic_keys[j]
            merged_tids = (dic[key1] & dic[key2])
            if len(merged_tids) >= min_support:
                out_dic[key1 | key2] = (dic[key1] & dic[key2])
        if out_dic:
            dic_list.append(out_dic)
        num_sets = num_sets + len(out_dic)
    return dic_list, num_sets


def partition_sets(dic_list, num_sets, num_partitions):
    partition_size = num_sets / num_partitions
    dic_partitions = [{} for i in range(num_partitions)]
    partition_index = 0
    for dic in dic_list:
        dic_partitions[partition_index].update(dic)
        if len(dic_partitions[partition_index]) > partition_size and partition_index < num_partitions - 1:
            partition_index = partition_index + 1

    return dic_partitions


def parallel_eclat(in_q, out_q):
    dic, min_support = in_q.get()

    dic_out, dic_list = eclat_seq.get_frequent_itemsets(dic, min_support)

    out_q.put(dic_out)
