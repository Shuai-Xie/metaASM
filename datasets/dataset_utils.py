import numpy as np
from pprint import pprint


def get_cls_img_idxs_dict(ds_labels, num_classes):
    """
    parse ds_labels into each cls img idxs
    @param ds_labels: dataset labels
    @param num_classes:
    @return: dict{ cls_idx : img_idxs_list,.. }
    """
    cls_img_idxs = {}
    for j in range(num_classes):
        # use iter idx, so not ori img idx
        cls_img_idxs[j] = [i for i, label in enumerate(ds_labels) if label == j]
    return cls_img_idxs


def print_dataset_info(name, dataset, num_classes):
    print(name, len(dataset))
    get_per_cls_sample_num(dataset.targets, num_classes)


def get_per_cls_sample_num(targets, num_classes):
    cls_num = {i: 0 for i in range(num_classes)}
    for i in targets:
        cls_num[i] += 1
    print(cls_num)


def get_per_cls_img_idxs(targets, num_classes):
    cls_img_idxs = {i: [] for i in range(num_classes)}
    for img_idx, cls in enumerate(targets):
        cls_img_idxs[cls].append(img_idx)
    pprint(cls_img_idxs)


def get_dataset_imb(targets, num_classes):
    cls_num = {i: 0 for i in range(num_classes)}
    for i in targets:
        cls_num[i] += 1
    vals = list(cls_num.values())
    imb = np.max(vals) / np.min(vals)
    return imb
