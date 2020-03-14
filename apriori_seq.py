def count_frequent_n_items(transaction_dict, item_count):
    for items in transaction_dict.values():
        for subset_items in item_count.keys():
            if subset_items.issubset(items):
                item_count[subset_items] += 1

    return item_count


def prune_count(item_count, min_support):
    item_count_prune = {item: count for item, count in item_count.items() if count >= min_support}

    return item_count_prune


def create_count_dict(item_count):
    item_count_out = {}
    unique_itemsets = set()
    for itemset in item_count.keys():
        unique_itemsets.update(itemset)

    for itemset in item_count.keys():
        for item in unique_itemsets:
            if item not in itemset:
                item_count_out[itemset.union(set([item]))] = 0

    return item_count_out
