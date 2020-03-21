import torch
import numpy as np

from datasets.cls_datasets import CIFAR
from datasets.imb_data_utils import get_cls_img_idxs_dict, transform_train
from utils.asm_utils import detect_unlabel_imgs, get_select_fn
import random

"""
sort each cls samples by criterion
"""


@torch.no_grad()
def sort_cls_samples(model, label_dataset, num_classes, criterion='lc'):
    # 每类图片 idxs
    cls_img_idxs = get_cls_img_idxs_dict(label_dataset.targets, num_classes)
    y_pred_prob = detect_unlabel_imgs(model, label_dataset.data, num_classes, bs=100)  # [N,10] prob vector

    sort_cls_idxs_dict = {}
    assert criterion in ['rs', 'lc', 'ms', 'en'], 'no such criterion'
    select_fn = get_select_fn(criterion)

    for cls_idx, img_idxs in cls_img_idxs.items():
        img_idxs = np.array(img_idxs)
        cls_probs = y_pred_prob[img_idxs]  # [n,10]
        # sorted idxs in list
        _, sort_cls_idxs = select_fn(cls_probs, n_samples=len(cls_probs))  # sort total
        # recover to total label idx
        sort_cls_idxs_dict[cls_idx] = img_idxs[sort_cls_idxs]

    return sort_cls_idxs_dict


def check_sample_targets(cls_idxs_dict, targets):
    for cls, img_idxs in cls_idxs_dict.items():
        print('class:', cls, [targets[i] for i in img_idxs])


"""
build meta dataset by different sampling methods
"""


def build_meta_dataset(label_dataset, idx_to_meta):
    random.shuffle(idx_to_meta)  # 原本各类按顺序

    meta_dataset = CIFAR(
        data=np.take(label_dataset.data, idx_to_meta, axis=0),
        targets=np.take(label_dataset.targets, idx_to_meta, axis=0),
        transform=transform_train
    )
    return meta_dataset


# random sample
def random_sample_meta_dataset(label_dataset, num_meta, num_classes):
    img_idxs = list(range(len(label_dataset.targets)))
    random.shuffle(img_idxs)
    idx_to_meta = img_idxs[:int(num_meta * num_classes)]

    return build_meta_dataset(label_dataset, idx_to_meta)


# random sample in a systematic way, loyal to original data distribution
def random_system_sample_meta_dataset(label_dataset, sort_cls_idxs_dict, num_meta, mid=None):  # 等距抽样
    idx_to_meta = []

    for cls, img_idxs in sort_cls_idxs_dict.items():  # 能处理各类 样本数量不同, list 不等长
        step = len(img_idxs) // num_meta
        mid = mid % step if mid else random.randint(0, step)  # 指定每个系统内 要取的元素位置
        idx_to_meta.extend([img_idxs[min(i * step + mid, len(img_idxs) - 1)]
                            for i in range(num_meta)])  # 等间隔

    return build_meta_dataset(label_dataset, idx_to_meta)


# sample best train on label_dataset
def sample_best_train_meta_dataset(label_dataset, sort_cls_idxs_dict, num_meta):
    idx_to_meta = []

    for cls, img_idxs in sort_cls_idxs_dict.items():
        idx_to_meta.extend(img_idxs[-num_meta:])  # 最可信的

    return build_meta_dataset(label_dataset, idx_to_meta)
