import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
from datasets.cls_datasets import CIFAR
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
        cls_img_idxs[j] = [i for i, label in enumerate(ds_labels) if label == j]
    return cls_img_idxs


normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

# Process PIL.Image / np img
transform_train = transforms.Compose([
    # pad, 4D input tensor, lrtb [可处理 1D,2D,3D 输入，分别对应不同 pad 输入]
    # same to np.pad, reflect 32x32 -> 40x40
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),  # 4D
                                      (4, 4, 4, 4), mode='reflect').squeeze()),  # recover 3D
    transforms.ToPILImage(),  # C x H x W tensor
    transforms.RandomCrop(32),  # 32x32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # will div(255) to [0,1]
    normalize
])


def build_dataset(dataset, num_meta):
    """
    @param dataset: dataset name
    @param num_meta: meta data num per class
    @return:
        train_data_meta, train_data, test_dataset
    """
    if dataset == 'cifar10':  # 5000/1000 * 10
        train_dataset = torchvision.datasets.CIFAR10(root='/nfs/xs/Datasets/CIFAR10', train=True, download=True,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('/nfs/xs/Datasets/CIFAR10', train=False,
                                                    transform=transform_test)
        img_num_list = [num_meta] * 10  # 10 cls, each cls has num_meta samples
        num_classes = 10

    elif dataset == 'cifar100':  # 500/100 * 100
        train_dataset = torchvision.datasets.CIFAR100(root='/nfs/xs/Datasets/CIFAR100', train=True, download=True,
                                                      transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100('/nfs/xs/Datasets/CIFAR100', train=False,
                                                     transform=transform_test)
        img_num_list = [num_meta] * 100
        num_classes = 100
    else:
        raise ValueError('no such dataset!')

    cls_img_idxs = get_cls_img_idxs_dict(train_dataset.targets, num_classes)

    # store train/meta img idxs, for np.delete
    idx_to_meta = []
    idx_to_train = []
    for cls_idx, img_idxs in cls_img_idxs.items():
        random.shuffle(img_idxs)
        img_num = img_num_list[int(cls_idx)]  # 10, num_meta
        idx_to_meta.extend(img_idxs[:img_num])  # front 10 as meta
        idx_to_train.extend(img_idxs[img_num:])  # remain as train

    # judge if random.seed(args.seed) works
    # print('meta idxs:', idx_to_meta)

    # note transform
    train_data_meta = CIFAR(
        data=np.take(train_dataset.data, idx_to_meta, axis=0),
        targets=np.take(train_dataset.targets, idx_to_meta, axis=0),
        transform=transform_train
    )
    train_data = CIFAR(
        data=np.take(train_dataset.data, idx_to_train, axis=0),
        targets=np.take(train_dataset.targets, idx_to_train, axis=0),
        transform=transform_train
    )

    return train_data_meta, train_data, test_dataset


def get_imb_num_list(dataset, imb_factor=None, num_meta=0):
    """
    @param dataset: cifar10, cifar100
    @param imb_factor: 10,20,100,...
    @param num_meta: to compute remain instances of each class
    @return:
    """
    if dataset == 'cifar10':
        img_max = 50000 / 10 - num_meta
        cls_num = 10
    elif dataset == 'cifar100':
        img_max = 50000 / 100 - num_meta
        cls_num = 100
    else:
        return ValueError('no such dataset!')

    if imb_factor is None:  # no imbalance, return base_num = img_max
        return [img_max] * cls_num

    # process imbalance
    imb_num_list = []
    imb_factor = 1 / imb_factor  # 从 img_max 开始乘
    for cls_idx in range(cls_num):
        # from class 0-9
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        imb_num_list.append(int(num))

    return imb_num_list


def get_imb_meta_test_datasets(dataset, num_classes, num_meta, imb_factor, split=None, ratio=1.):
    """
    :param dataset: cifar10, cifar100
    :param num_classes: 10, 100
    :param num_meta: 10
    :param imb_factor: 1,10,20,50,100
    :param split: asm initial_annotated_ratio
    :param ratio: [0,1] 取多少数据参与训练
    :return:
    """
    train_meta_dataset, train_dataset, test_dataset = build_dataset(dataset, num_meta)

    # used image num of each class by imb_factor
    imb_num_list = get_imb_num_list(dataset, imb_factor, num_meta)
    # print('imb_img_num:', imb_num_list)
    # [5000, 3871, 2997, 2320, 1796, 1391, 1077, 834, 645, 500]

    # get img idxs of each class
    data_list = get_cls_img_idxs_dict(train_dataset.targets, num_classes)

    # 构造不平衡数据集
    idx_to_use = []
    for cls_idx, img_idxs in data_list.items():
        random.shuffle(img_idxs)  # shuffle each cls img_idxs
        img_num = int(imb_num_list[int(cls_idx)] * ratio)
        idx_to_use.extend(img_idxs[:img_num])

    # build imbalance dataset
    imb_train_dataset = CIFAR(
        data=np.take(train_dataset.data, idx_to_use, axis=0),
        targets=np.take(train_dataset.targets, idx_to_use, axis=0),
        transform=transform_train
    )

    if split and 0 < split <= 1:  # for asm
        data_list = get_cls_img_idxs_dict(imb_train_dataset.targets, num_classes)
        label_idxs = []
        for cls_idx, img_idxs in data_list.items():
            random.shuffle(img_idxs)
            img_num = int(len(img_idxs) * split)
            label_idxs.extend(img_idxs[:img_num])

        # label/unlabel 均服从 imb_factor
        # label, train on this, so use transform_train
        label_dataset = CIFAR(
            data=np.take(imb_train_dataset.data, label_idxs, axis=0),
            targets=np.take(imb_train_dataset.targets, label_idxs, axis=0),
            transform=transform_train
        )
        # unlabel, detect on this, so use transform_test
        unlabel_dataset = CIFAR(
            data=np.delete(imb_train_dataset.data, label_idxs, axis=0),
            targets=np.delete(imb_train_dataset.targets, label_idxs, axis=0),
            transform=None  # todo: to transform, just get data
        )

        print_dataset_info('label_dataset:', label_dataset, num_classes)
        print_dataset_info('unlabel_dataset:', unlabel_dataset, num_classes)
        print_dataset_info('train_meta_dataset:', train_meta_dataset, num_classes)
        print_dataset_info('test_dataset:', test_dataset, num_classes)

        return label_dataset, unlabel_dataset, train_meta_dataset, test_dataset

    else:
        print_dataset_info('imb_train_dataset:', imb_train_dataset, num_classes)
        print_dataset_info('train_meta_dataset:', train_meta_dataset, num_classes)
        print_dataset_info('test_dataset:', test_dataset, num_classes)

        return imb_train_dataset, train_meta_dataset, test_dataset


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


def test_imb_sample_num():
    imb_num = get_imb_num_list('cifar10', imb_factor=1, num_meta=0)
    print(imb_num)
    print(sum(imb_num))  # 123


def test_split_dataloader(ratio=1.):
    dataset, num_classes = 'cifar10', 10
    label_dataset, unlabel_dataset, train_meta_dataset, test_dataset = \
        get_imb_meta_test_datasets(dataset, num_classes, num_meta=0, imb_factor=10, split=0.1, ratio=ratio)


if __name__ == '__main__':
    # test_imb_sample_num()
    # test_split_dataloader()
    # test_split_dataloader(ratio=0.1)
    build_dataset('cifar10', 10)
