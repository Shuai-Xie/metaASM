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
from optim.adabound import AdaBound

from datasets.cifar.dataset import CIFAR
from datasets.build_imb_dataset import get_imb_meta_test_datasets

from net import *
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
    '--num_meta', '10',  # 从 labelset 选出的代表性 sample 数量，恰好 = bs?
    '-K', '50',  # 1/2 bs, batch uncertain samples, will rm hc in uc, so <= 50
    '-uc', 'lc',
    '--split', '0.4',
    '--ratio', '1',  # 小样本
    '--init_epochs', '20',  # meta_epochs, finetune_epochs?
    '--epochs', '50',
    '--ckpt', 'output/asm_meta_cifar10_imb1_s0.4_r1.0_Mar19_202659/rs32_epoch_19.pth',
    '--tag', 'asm'
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
unlabel_loader = DataLoader(unlabel_dataset,  # DataLoader will cvt np to tensor
                            batch_size=args.batch_size,
                            drop_last=False,  # todo: False 查看 unlabel 数量
                            shuffle=False, **kwargs)  # shuffle 能带来更稳定的 imb
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)


def valid_save_model(epoch, model, vnet=None):
    global best_prec1, best_epoch, best_model

    _, prec1 = validate(test_loader, model, criterion,
                        epoch, args.print_freq, writer)

    if prec1 > best_prec1:
        best_prec1, best_epoch = prec1, epoch
        best_model = model
        save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(epoch)),
                   model, epoch, best_prec1)
        if vnet:
            save_model(os.path.join(model_save_dir, 'vnet_epoch_{}.pth'.format(epoch)),
                       vnet, epoch, best_prec1)
        print(f'epoch {epoch}, best acc: {best_prec1}, best epoch: {best_epoch}')

    return best_model, best_prec1, best_epoch


def train_init_epochs(model, init_epochs):
    global best_prec1, best_epoch, best_model

    for epoch in range(init_epochs):
        # train
        train_base(label_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer)

        best_model, best_prec1, best_epoch = valid_save_model(epoch, model)

    print(f'Finish init train, best acc: {best_prec1}, best epoch: {best_epoch}')

    return best_model, best_prec1, best_epoch


"""
batch asm with meta dataset

steps
1. train init_epochs on D_L
2. infer on D_L, get P_L
3. train meta_epochs on {D_M, D_U}
    for B_U in D_U
        infer on B_U, get {B_hc, B_uc}
        get D_M by system-sampling on D_L
        form B_asm = {D_M, B_hc, B_uc}
        train n=1 iterations on B_asm
    update P_L
    collect {D_hc, D_uc}
4. form D_finetune = {D_L, D_hc, D_uc}
5. train finetune_epochs on D_finetune

注意：

1. meta_epochs 中 unlabel_loader 每轮采用不同 shuffle 方式
    记录 global_uc_samples 后，发现 SGD 比 Adam 要使用更多 uc_sample，因为前期收敛慢

2. num_meta 数量会使得训练时长翻倍，因为直接影响 asm_meta_loader 训练的 batch 数
"""

# exp
os.makedirs('exp', exist_ok=True)
exp = f'{args.tag}_{args.dataset}_imb{args.imb_factor}_s{args.split}_r{args.ratio}_m{args.num_meta}_{get_curtime()}'
dump_json(vars(args), out_path=f'exp/{exp}.json')
print('exp:', exp)

# tensorboard / log
writer = SummaryWriter(log_dir=os.path.join('runs', exp))
sys.stdout = Logger(f'exp/{exp}.log', sys.stdout)

# output model
model_save_dir = os.path.join('output', exp)
os.makedirs(model_save_dir, exist_ok=True)

if __name__ == '__main__':

    # build model
    model = ResNet32(args.num_classes).cuda()
    # vnet = VNet(1, 100, 1).cuda()  # weights 还输入 loss
    print('build model done!')

    # SGD
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    # Adam
    # optimizer_a = torch.optim.Adam(model.params(),
    #                                lr=1e-3,
    #                                betas=(0.9, 0.999), eps=1e-8)

    criterion = nn.CrossEntropyLoss().cuda()
    uc_select_fn = get_select_fn(args.uncertain_criterion)

    best_prec1, best_epoch, best_model = 0, 0, None

    # train on initial labelset
    if args.ckpt is not None:
        print('load pretrain model')
        model, optimizer_a, best_prec1, best_epoch = load_model(model, args.ckpt, optimizer_a)
        writer.add_scalar('Test/top1_acc', best_prec1, global_step=best_epoch)
    else:
        print('train on initial labelset')
        model, best_prec1, best_epoch = train_init_epochs(model, args.init_epochs)
    # return best model

    # select meta data
    random.seed()
    # valid_loader = DataLoader(meta_dataset,
    #                           batch_size=args.batch_size,
    #                           drop_last=False,
    #                           shuffle=True, **kwargs)
    meta_epochs = 50

    global_total_uc_idxs = []

    # epoch 29 reduce lr [20, 50]
    for epoch in range(best_epoch + 1, meta_epochs):
        # todo: early adjust lr
        adjust_lr(args.lr, optimizer_a, epoch, writer)

        # update sample probs
        # 每次都要在 label_dataset 重新推理计算 uncertain 排名
        sort_cls_idxs_dict = sort_cls_samples(model, label_dataset, args.num_classes, criterion='lc')

        begin_step = epoch * len(unlabel_loader)

        total_uc_idxs = []

        # img_idxs get the identical_img_idx of each unlabel sample
        for i, (input, target, img_idxs) in enumerate(unlabel_loader):
            # fetch asm data from unlabel loader
            results = asm_split_hc_delta_uc_K(model, input, target,
                                              args.delta,
                                              uc_select_fn, args.uncertain_samples_size)

            # gloal uc_idxs <- local uc_idxs
            # 虽然 uc_ratio 从全局来看 < 50%，但尚不确定是否在不同 epoch 有不同的 uc_sample 选出
            batch_uc_idxs = cvt_iter_to_list(img_idxs[results['uc']['idxs']], int)
            total_uc_idxs.extend(batch_uc_idxs)

            writer.add_scalars(f'ASM/ratios', {
                'hc_ratio': results['hc']['ratio'],
                'uc_ratio': results['uc']['ratio']
            }, global_step=begin_step + i)

            writer.add_scalars(f'ASM/accs', {
                'hc_acc': results['hc']['acc'],
                'uc_acc': results['uc']['acc']
            }, global_step=begin_step + i)

            # select subset from labelset, 系统抽样 all-hard levels
            meta_dataset = random_system_sample_meta_dataset(label_dataset,
                                                             sort_cls_idxs_dict, args.num_meta)

            asm_data = np.append(results['hc']['data'], results['uc']['data'], axis=0)
            asm_targets = np.append(results['hc']['targets'], results['uc']['targets'], axis=0)

            asm_meta_dataset = CIFAR(
                data=np.append(meta_dataset.data, asm_data, axis=0),
                targets=np.append(meta_dataset.targets, asm_targets, axis=0),
                transform=transform_train
            )
            asm_meta_loader = DataLoader(asm_meta_dataset,
                                         batch_size=args.batch_size,  # 保持和原模型相同 batchsize
                                         drop_last=False,
                                         shuffle=True, **kwargs)

            meta_losses, meta_accs = [], []
            for _ in range(1):  # meta_train 在同一批次训练多次反而不好!
                meta_loss, meta_acc = train_meta(asm_meta_loader, model, criterion,
                                                 optimizer_a,
                                                 step=f'{i}/{len(unlabel_loader)}', print_freq=1)
                meta_losses.append(meta_loss)
                meta_accs.append(meta_acc)

            writer.add_scalar('ASM/train_loss', np.mean(meta_losses), global_step=begin_step + i)
            writer.add_scalar('ASM/train_acc', np.mean(meta_accs), global_step=begin_step + i)

        # 获取每个 asm epoch 之后并集，为全局使用的 uc_samples
        global_total_uc_idxs = list(set(global_total_uc_idxs) | set(total_uc_idxs))

        writer.add_scalars('ASM/total_uc', {  # 理想状态
            'current': len(total_uc_idxs),  # 持续下降
            'global': len(global_total_uc_idxs)  # 保持稳定
        }, global_step=epoch)

        # valid after each asm bach
        valid_save_model(epoch, model)

    print(f'Finish train, best acc: {best_prec1}, best epoch: {best_epoch}')
