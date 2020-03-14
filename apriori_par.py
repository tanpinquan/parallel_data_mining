import apriori_seq


def partition_sets(transaction_dict, num_partitions):
    num_transactions = len(transaction_dict)
    partition_size = num_transactions / num_partitions
    transaction_partitions = [{} for i in range(num_partitions)]
    partition_index = 0
    for transaction, items in transaction_dict.items():
        transaction_partitions[partition_index][transaction] = items
        if len(transaction_partitions[partition_index]) > partition_size and partition_index < num_partitions - 1:
            partition_index = partition_index + 1

    return transaction_partitions


def parallel_apriori(in_q, out_q):
    transaction_dict, count_dict = in_q.get()

    count_dict = apriori_seq.count_frequent_n_items(transaction_dict, count_dict)
    # print(count_dict)
    out_q.put(count_dict)


def merge_proc_counts(count_list):
    # out_count = {}
    out_count = {item: 0 for item in count_list[0].keys()}

    for count_dict in count_list:
        for item, counts in count_dict.items():
            out_count[item] += counts

    return out_count
