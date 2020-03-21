import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
import random
import numpy as np
import copy

from datasets.cls_datasets import CIFAR
from datasets.imb_data_utils import *
from net.resnet import ResNet32
from optim.adabound import AdaBound

from engine import *
from utils import *

parser = argparse.ArgumentParser(description='Classification on cifar10')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
# parser.add_argument('--net', default='ResNet18', type=str,
#                     help='available models in net/classifier dir')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',  # train bs
                    help='input batch size for training (default: 100)')
parser.add_argument('--num_classes', type=int, default=10)  # cifar10
parser.add_argument('--num_meta', type=int, default=10,  # meta data, 10 samples/class
                    help='The number of meta data for each class. 此 meta 非彼 meta')
parser.add_argument('--imb_factor', type=int, default=10)  # imbalance factor
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',  # total 100 epoches
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,  # init lr=0.1
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')  # momentum=0.9
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,  # decay
                    help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default=42, metavar='S',  # random
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--tag', default='exp', type=str,
                    help='experiment tag to create tensorboard, model save dir name')
# asm
parser.add_argument('--init_epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train on init trainset')
parser.add_argument('--meta_epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train on init trainset')
parser.add_argument('--finetune_epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train on init trainset')
parser.add_argument('--ckpt', type=str,
                    help='already pretrained model')
parser.add_argument('--split', default=0.1, type=float,
                    help='Initial Annotated Samples Ratio. default: 0.1')
parser.add_argument('--ratio', default=1.0, type=float,
                    help='use how many samples of each cls')
parser.add_argument('-K', '--uncertain_samples_size', default=1000, type=int,
                    help='Uncertain samples selection size. default: 1000')
parser.add_argument('-uc', '--uncertain_criterion', default='ms', type=str,
                    help='Uncertain selection Criteria:\n'
                         'rs(Random Sampling)\n'
                         'lc(Least Confidence)\n'
                         'ms(Margin Sampling)\n'
                         'en(Entropy)')
parser.add_argument('--delta', default=0.5, type=float,  # prob > 0.9
                    help="High confidence samples selection threshold. default: 0.05")

params = [
    '--dataset', 'cifar10',
    '--num_classes', '10',
    '--imb_factor', '1',
    # important params
    '--num_meta', '50',  # 从 labelset 选出的代表性 sample 数量，恰好 = bs?
    '-K', '50',  # 1/2 bs, batch uncertain samples, will rm hc in uc, so <= 50
    '--split', '0.4',
    '--ratio', '1',  # 小样本
    '--init_epochs', '20',  # meta_epochs, finetune_epochs?
    '--meta_epochs', '10',
    '--finetune_epochs', '10',  # lr decay
    # '--ckpt', 'output/meta_cifar10_imb1_s0.4_r1.0_Mar17_115921/rs32_epoch_8.pth',
    '--tag', 'hope3'
]
args = parser.parse_args(params)
pprint(vars(args))

cudnn.benchmark = True

# get datasets
random.seed(args.seed)  # 保证初始划分 label/unlabel 一致
label_dataset, unlabel_dataset, _, test_dataset = get_imb_meta_test_datasets(
    args.dataset, args.num_classes, 0, args.imb_factor, args.split, args.ratio
)  # CIFAR10

# imb_train/valid_meta/test
kwargs = {'num_workers': 4, 'pin_memory': True}

# 将 label/unlabel loader 放入同一 batch
label_loader = DataLoader(label_dataset,
                          batch_size=args.batch_size,
                          drop_last=False,
                          shuffle=True, **kwargs)
unlabel_loader = DataLoader(unlabel_dataset,
                            batch_size=args.batch_size,
                            drop_last=False,
                            shuffle=True, **kwargs)  # shuffle 能带来更稳定的 imb
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)

"""steps
1. train init_epochs on labelset
2. infer on labelset, select typical samples by prob distribution as meta data
3. infer on unlabelset [unlabel_loader] [每个 batch 就是个子任务？最后 fine_tune?]
    3.1 infer on batch unlabel data -> {hc, uc, metadata} -> meta_loader
    3.2 build & train on meta_loader
    Finish one epoch
    3.3 update label_loader / unlabel_loader
    3.4 resample meta data with new labelset
    if avg_hc_ratio > 0.9, goto 5
4. goto 3
5. finetune
"""


def empty_x():
    return np.empty([0] + list(label_dataset.data.shape[1:]), dtype='uint8')  # (0,32,32,3)


def empty_y():
    return np.empty([0], dtype='int64')


def train_init_epochs():
    global best_prec1
    best_epoch = 0
    for epoch in range(args.init_epochs):
        # train
        train_base(label_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer)
        # evaluate on testset
        prec1 = evaluate(test_loader, model, criterion,
                         epoch, args.print_freq, writer)
        # remember best prec@1 and save checkpoint
        if prec1 > best_prec1:
            best_prec1, best_epoch = prec1, epoch
            save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(epoch)),
                       model, epoch, best_prec1)
    return best_prec1, best_epoch


if __name__ == '__main__':
    exp = f'{args.tag}_{args.dataset}_imb{args.imb_factor}_s{args.split}_r{args.ratio}_{get_curtime()}'
    print('exp:', exp)

    model = ResNet32(args.num_classes).cuda()
    print('build model done!')

    # SGD
    # optimizer_a = torch.optim.SGD(model.params(), args.lr,
    #                               momentum=args.momentum, nesterov=args.nesterov,
    #                               weight_decay=args.weight_decay)

    # Adam
    optimizer_a = torch.optim.Adam(model.params(),
                                   lr=1e-3,
                                   betas=(0.9, 0.999), eps=1e-8)

    # AdaBound
    # optimizer_a = AdaBound(model.params())

    criterion = nn.CrossEntropyLoss().cuda()

    select_fn = get_select_fn(args.uncertain_criterion)

    writer = SummaryWriter(log_dir=os.path.join('runs', exp))
    model_save_dir = os.path.join('output', exp)
    os.makedirs(model_save_dir, exist_ok=True)

    best_prec1, best_epoch = 0, 0

    # [1] train on initial labelset
    if args.ckpt:
        print('load pretrain model')
        model, optimizer_a, best_prec1, best_epoch = load_model(model, args.ckpt, optimizer_a)
        begin_epoch = best_epoch + 1
        writer.add_scalar('Test/top1_acc', best_prec1, global_step=best_epoch)
    else:
        print('train on initial labelset')
        best_prec1, best_epoch = train_init_epochs()
        begin_epoch = args.init_epochs

    print(f'Finish initial train, best acc: {best_prec1}, best epoch: {best_epoch}')

    # [2] build meta dataset by system-sampling on label_dataset by probs
    # criterion 可自定义
    random.seed()  # 随机
    sort_cls_idxs_dict = sort_cls_samples(model, label_dataset, args.num_classes,
                                          criterion=args.uncertain_criterion)
    # [3] train on unlabel data & meta data in an asm manner
    print('train on unlabel data & meta data')

    total_label_data, total_label_targets = empty_x(), empty_y()
    total_unlabel_data, total_unlabel_targets = empty_x(), empty_y()

    asm_batch_loaders = []

    # todo: just train one epoch, make it efficient!
    # 如果这种方式，evaluate 得到较好的结果，就成功了
    print('epoch:', begin_epoch)

    # 与 labelset 中代表性样本一起 训练 model，并选出困难样本! 一个 epoch 做完
    for i, (input, target) in enumerate(unlabel_loader):
        # infer on batch unlabel data -> {hc, uc}
        with torch.no_grad():
            input_var = to_var(input, requires_grad=False)
            outputs = model(input_var)
            probs = F.softmax(outputs, dim=-1)
            probs = to_numpy(probs)

        # recover to np
        input_data = to_numpy(input_var).transpose((0, 2, 3, 1)).astype('uint8')
        target_val = to_numpy(target).astype('int64')

        # hc
        hc_idxs, hc_preds = get_high_conf_samples(probs, args.delta)
        hc_ratio = len(hc_idxs) / len(probs)
        writer.add_scalar('Meta/hc_ratio', hc_ratio, global_step=i)
        if len(hc_idxs) > 0:
            hc_gts = np.take(target_val, hc_idxs, axis=0)
            hc_acc = sum(hc_preds == hc_gts) / len(hc_gts)
            writer.add_scalar('Meta/hc_acc', hc_acc, global_step=i)

        # uc
        _, uc_idxs = select_fn(probs, args.uncertain_samples_size)
        # uc_idxs = [k for k in uc_idxs if k not in hc_idxs]  # rm uc in hc
        writer.add_scalars('Meta/samples', {
            'uc': len(uc_idxs),
            'hc': len(hc_idxs)
        }, global_step=i)

        # meta label data
        meta_cls_idxs_dict = random_system_sample(sort_cls_idxs_dict, args.num_meta, mid=i)  # 指定遍历
        meta_dataset = build_meta_dataset(meta_cls_idxs_dict, label_dataset)

        # train on asm_meta_dataset
        # todo: 手工选固定个数的，考虑 snet
        asm_meta_dataset = CIFAR(
            data=np.append(meta_dataset.data,  # 不用 hc, 模型不行 hc_acc 太低
                           np.take(input_data, uc_idxs, axis=0), axis=0),
            targets=np.append(meta_dataset.targets, target_val[uc_idxs], axis=0),
            transform=transform_train
        )
        imb = get_dataset_imb(asm_meta_dataset.targets, args.num_classes)
        writer.add_scalar('Meta/imb', imb, global_step=i)

        # todo: 使用 meta_model 训练? 多次 epoch 迭代后，不好的样本作为 hard?
        asm_meta_loader = DataLoader(asm_meta_dataset,
                                     batch_size=args.batch_size,
                                     drop_last=False,
                                     shuffle=True, **kwargs)
        meta_loss, meta_acc = train_meta(asm_meta_loader, model, criterion,
                                         optimizer_a,
                                         step=i, print_freq=1)

        writer.add_scalars('Meta/loss', {
            f'epoch_{begin_epoch}': meta_loss
        }, global_step=i)

        writer.add_scalars('Meta/acc', {
            f'epoch_{begin_epoch}': meta_acc
        }, global_step=i)

        # update label/unlabel data
        total_label_data = np.append(label_dataset.data,
                                     np.take(input_data, uc_idxs, axis=0), axis=0)
        total_label_targets = np.append(label_dataset.targets,
                                        np.take(target_val, uc_idxs, axis=0), axis=0)
        total_unlabel_data = np.append(total_unlabel_data,
                                       np.delete(input_data, uc_idxs, axis=0), axis=0)
        total_unlabel_targets = np.append(total_unlabel_targets,
                                          np.delete(target_val, uc_idxs, axis=0), axis=0)

    # eval
    prec1 = evaluate(test_loader, model, criterion,
                     begin_epoch, args.print_freq, writer)

    if prec1 > best_prec1:
        best_prec1, best_epoch = prec1, begin_epoch
        save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(best_epoch)),
                   model, best_epoch, best_prec1)

    # todo: 既然 data 已经确定，为什么不直接 shuffle?
    # 如果保留 batch 单位，模型性能不会提升
    label_dataset = CIFAR(
        data=total_label_data,
        targets=total_label_targets,
        transform=transform_train
    )
    label_loader = DataLoader(label_dataset,
                              batch_size=args.batch_size,
                              drop_last=False,
                              shuffle=True, **kwargs)

    for epoch in range(begin_epoch + 1, args.epochs):
        # train
        train_base(label_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer)
        # eval
        prec1 = evaluate(test_loader, model, criterion,
                         epoch, args.print_freq, writer)

        if prec1 > best_prec1:
            best_prec1, best_epoch = prec1, epoch
            save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(best_epoch)),
                       model, best_epoch, best_prec1)

    print('Best accuracy: {}, epoch: {}'.format(best_prec1, best_epoch))
