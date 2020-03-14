from collections import OrderedDict


def prune_dict(dic, min_support):
    out_dic = {}
    for key, value in dic.items():
        if len(value) >= min_support:
            out_dic[key] = value

    return out_dic


def merge_tids(dic):
    out_dic = {}

    for key, value in dic.items():
        for key2, value2 in dic.items():
            if key != key2:
                # out_dic[frozenset([key, key2])] = (value & value2)
                out_dic[key | key2] = (value & value2)

    return out_dic


def merge_tids_2(dic, min_support):
    out_dic = {}
    dic_keys = list(dic.keys())

    for i in range(len(dic_keys)):
        for j in range(i + 1, len(dic_keys)):
            key1 = dic_keys[i]
            key2 = dic_keys[j]
            merged_tids = (dic[key1] & dic[key2])
            if len(merged_tids) >= min_support:
                out_dic[key1 | key2] = (dic[key1] & dic[key2])
    return out_dic


def get_frequent_itemsets(dic, min_support):
    dic_list = [dic]
    out_dic = dic.copy()

    i = 0
    while len(dic_list[i]) != 0:
        dic_list.append(merge_tids_2(dic_list[i], min_support))
        i = i + 1
        out_dic.update(dic_list[i])

    return out_dic, dic_list
