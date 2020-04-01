import torchvision
import random
from datasets.cifar.dataset import CIFAR, CIFAR_unlabel
from datasets.dataset_utils import *
from datasets.transforms import *


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

    # original img idxs
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
    # here! idx is already not the original idx
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
        unlabel_dataset = CIFAR_unlabel(
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


def test_imb_sample_num():
    imb_num = get_imb_num_list('cifar10', imb_factor=1, num_meta=0)
    print(imb_num)
    print(sum(imb_num))  # 123


def test_unlabel_dataloader(ratio=1.):
    dataset, num_classes = 'cifar10', 10
    label_dataset, unlabel_dataset, train_meta_dataset, test_dataset = \
        get_imb_meta_test_datasets(dataset, num_classes, num_meta=0, imb_factor=10, split=0.1, ratio=ratio)

    from torch.utils.data import DataLoader
    import torch

    unlabel_loader = DataLoader(unlabel_dataset,  # DataLoader will cvt np to tensor
                                batch_size=10,
                                drop_last=False,
                                shuffle=True)  # shuffle 能带来更稳定的 imb

    for epoch in range(3):
        torch.manual_seed(42)  # 外 loader 人工随机，复现

        # 只要 unlabel_loader 不更新，img_idx 相对与每张图片都是唯一的
        for i, (imgs, targets, img_idxs) in enumerate(unlabel_loader):
            print(f'{epoch}.{i}')
            # print(imgs.size(), imgs.dtype)
            print('img_idxs:', img_idxs)
            print('targets:', targets)

            # 内 loader 真正随机；如果不设置，每次训练 batch 都一样
            # 直接使用 torch.seed() 会有越界异常
            torch.manual_seed(random.randint(0, 100))  # 只指定0也不行

            for inner_epoch in range(2):
                for k, (imgs, targets, img_idxs) in enumerate(unlabel_loader):
                    print(f'-> {inner_epoch}.{k}')
                    print('img_idxs:', img_idxs)
                    print('targets:', targets)
                    break
            break


if __name__ == '__main__':
    # test_imb_sample_num()
    # test_split_dataloader()
    test_unlabel_dataloader(ratio=0.1)
