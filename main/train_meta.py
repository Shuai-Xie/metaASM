import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.1"

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

from datasets.cifar.dataset import CIFAR
from datasets.build_imb_dataset import get_imb_meta_test_datasets

from net import *
from optim import AdaBound
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
parser.add_argument('--ckpt_asm', type=str,
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
parser.add_argument('--delta', default=0.5, type=float,  # prob > 0.9, 越大 hc_acc 越低
                    help="High confidence samples selection threshold. default: 0.05")

"""
batch 内每次引入 新样本数目有限
"""

params = [
    '--dataset', 'cifar10',
    '--num_classes', '10',
    '--imb_factor', '1',
    # important params
    '--num_meta', '5',  # 从 labelset 选出的代表性 sample 数量，恰好 = bs?
    '-K', '50',  # todo: 根据 init_ratio 自动调节 k 值？或者将 hc 剩下的全部作为 k
    '-uc', 'lc',
    '--split', '0.4',
    '--ratio', '1',  # 小样本
    '--init_epochs', '30',  # meta_epochs, finetune_epochs?
    '--ckpt', 'output/hope_cifar10_imb1_s0.4_r1.0_m50_Mar23_201207/rs32_epoch_29.pth',
    '--tag', 'meta'
]
args = parser.parse_args(params)
pprint(vars(args))

cudnn.benchmark = True

# get datasets
random.seed(args.seed)  # 保证初始划分 label/unlabel 一致
label_dataset, unlabel_dataset, _, test_dataset = get_imb_meta_test_datasets(
    args.dataset, args.num_classes, 0, args.imb_factor, args.split, args.ratio
)  # CIFAR10
random.seed()  # 解除

# imb_train/valid_meta/test
kwargs = {'num_workers': 4, 'pin_memory': True}

# 将 label/unlabel loader 放入同一 batch
label_loader = DataLoader(label_dataset,
                          batch_size=args.batch_size,
                          drop_last=False,
                          shuffle=True, **kwargs)
unlabel_loader = DataLoader(unlabel_dataset,  # DataLoader will cvt np to tensor
                            batch_size=args.batch_size,
                            drop_last=False,
                            shuffle=True, **kwargs)  # shuffle 能带来更稳定的 imb
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)


def valid_save_model(epoch, model, vnet=None, prefix='Test'):
    global best_prec1, best_epoch, best_model

    _, prec1 = validate(test_loader, model, criterion,
                        epoch, args.print_freq, writer, prefix)

    if prec1 > best_prec1:
        best_prec1, best_epoch = prec1, epoch
        best_model = model
        save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(epoch)),
                   model, epoch, best_prec1)
        if vnet:
            save_model(os.path.join(model_save_dir, 'vnet_epoch_{}.pth'.format(epoch)),
                       vnet, epoch, best_prec1)
        print(f'{prefix} epoch {epoch}, best acc: {best_prec1}, best epoch: {best_epoch}')

    return best_model, best_prec1, best_epoch


def train_init_epochs(model, init_epochs, prefix='Init'):
    global best_prec1, best_epoch, best_model

    for epoch in range(init_epochs):
        # train
        train_base(label_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer, prefix=prefix)

        best_model, best_prec1, best_epoch = valid_save_model(epoch, model, prefix=prefix)

    return best_model, best_prec1, best_epoch


def train_asm_epochs(model, meta_epochs, inner_meta_epochs=None, prefix='ASM'):
    global best_prec1, best_epoch, best_model

    global_total_uc_idxs = []

    start_epoch = best_epoch + 1

    batch0_uc_idxs = {}

    for idx, epoch in enumerate(range(start_epoch, start_epoch + meta_epochs)):
        # adjust_lr(args.lr, optimizer_a, epoch, writer)

        if inner_meta_epochs is None:
            inner_meta_epochs = max(meta_epochs - idx, 1)

        # update sample probs 这部分比较花时间，比得上 test 1次了
        sort_cls_idxs_dict = sort_cls_samples(model, label_dataset, args.num_classes,
                                              criterion=args.uncertain_criterion)
        top_hard_meta_dataset = sample_top_hard_meta_dataset(label_dataset,
                                                             sort_cls_idxs_dict, args.num_meta)

        begin_step = idx * len(unlabel_loader)

        total_uc_idxs = []
        total_hc_ratios = []

        torch.manual_seed(args.seed)  # 保证每个 meta_epoch shuffle 方式一样

        for i, (input, target, img_idxs) in enumerate(unlabel_loader):
            # fetch asm data from unlabel loader
            # batch 没有改变，但是选出的样本不一样了!!
            results = asm_split_hc_delta_uc_K(model, input, target,
                                              args.delta,
                                              uc_select_fn, args.uncertain_samples_size)

            # gloal uc_idxs <- local uc_idxs
            # 虽然 uc_ratio 从全局来看 < 50%，但尚不确定是否在不同 epoch 有不同的 uc_sample 选出
            batch_uc_idxs = cvt_iter_to_list(img_idxs[results['uc']['idxs']], type=int)
            total_uc_idxs.extend(batch_uc_idxs)

            writer.add_scalars(f'{prefix}/ratios', {
                'hc_ratio': results['hc']['ratio'],
                'uc_ratio': results['uc']['ratio']
            }, global_step=begin_step + i)

            writer.add_scalars(f'{prefix}/accs', {
                'hc_acc': results['hc']['acc'],
                'uc_acc': results['uc']['acc']
            }, global_step=begin_step + i)

            if i % 2 == 0:
                meta_dataset = random_system_sample_meta_dataset(label_dataset,
                                                                 sort_cls_idxs_dict, args.num_meta)
            else:
                meta_dataset = top_hard_meta_dataset

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
            for k in range(inner_meta_epochs):
                torch.manual_seed(random.randint(0, 10000))
                meta_loss, meta_acc = train_meta(asm_meta_loader, model, criterion,
                                                 optimizer_a,
                                                 step=f'{i + 1}.{k + 1}/{len(unlabel_loader)}',
                                                 print_freq=1)  # 顶多为2
                meta_losses.append(meta_loss)
                meta_accs.append(meta_acc)

            writer.add_scalar(f'{prefix}/train_loss', np.mean(meta_losses), global_step=begin_step + i)
            writer.add_scalar(f'{prefix}/train_acc', np.mean(meta_accs), global_step=begin_step + i)

        # 获取每个 asm epoch 之后并集，为全局使用的 uc_samples
        global_total_uc_idxs = list(set(global_total_uc_idxs) | set(total_uc_idxs))

        writer.add_scalars(f'{prefix}/total_uc', {  # 理想状态
            'current': len(total_uc_idxs),  # 持续下降
            'global': len(global_total_uc_idxs)  # 保持稳定
        }, global_step=epoch)

        # valid after each asm epoch
        # 注意返回 best_model，如果返回 model，会出现 eval NoneType
        best_model, best_prec1, best_epoch = valid_save_model(epoch, model, prefix=prefix)

        # Early stop
        if np.mean(total_hc_ratios) > 0.8:
            break

    dump_json(batch0_uc_idxs, f'{exp_dir}/batch0_uc_idxs.json')

    # get total hc
    global_total_hc_idxs = list(set(unlabel_dataset.img_idxs) - set(global_total_uc_idxs))

    return best_model, best_prec1, best_epoch, global_total_uc_idxs, global_total_hc_idxs


# exp
exp = f'{args.tag}_{args.dataset}_imb{args.imb_factor}_s{args.split}_r{args.ratio}_m{args.num_meta}_{get_curtime()}'
exp_dir = f'exp/{exp}'
os.makedirs(exp_dir, exist_ok=True)
dump_json(vars(args), out_path=f'{exp_dir}/config.json')
print('exp:', exp)

# tensorboard / log
writer = SummaryWriter(log_dir=os.path.join('runs', exp))
sys.stdout = Logger(f'{exp_dir}/run.log', sys.stdout)

# output model
model_save_dir = os.path.join('output', exp)
os.makedirs(model_save_dir, exist_ok=True)

if __name__ == '__main__':
    # build model
    model = ResNet32(args.num_classes).cuda()
    # vnet = VNet(1, 100, 1).cuda()  # weights 还输入 loss
    print('build model done!')

    # SGD
    # optimizer_a = torch.optim.SGD(model.params(), args.lr,
    #                               momentum=args.momentum, nesterov=args.nesterov,
    #                               weight_decay=args.weight_decay)

    # Adam
    optimizer_a = torch.optim.Adam(model.params(),
                                   lr=1e-3,
                                   betas=(0.9, 0.999), eps=1e-8)

    criterion = nn.CrossEntropyLoss().cuda()
    uc_select_fn = get_select_fn(args.uncertain_criterion)

    best_prec1, best_epoch, best_model = 0, 0, None

    ## init train
    if args.ckpt is not None:
        print('load init_trained model')
        best_model, optimizer_a, best_prec1, best_epoch = load_model(model, args.ckpt, optimizer_a)
        writer.add_scalar('Init/test_acc', best_prec1, global_step=best_epoch)
    else:
        print(f'begin {args.init_epochs} epochs initial train')
        # return model is best_model
        best_model, best_prec1, best_epoch = train_init_epochs(model, args.init_epochs, prefix='Init')

    print(f'Finish init train, best acc: {best_prec1}, best epoch: {best_epoch}')

    model = best_epoch

    ## asm meta train
    if args.ckpt_asm is not None:
        print('load asm_trained model')
        best_model, optimizer_a, best_prec1, best_epoch = load_model(model, args.ckpt_asm, optimizer_a)
        asm_result = load_json(f'exp/{args.ckpt_asm.split("/")[1]}/asm_result.json')
        global_total_uc_idxs = asm_result['global_total_uc_idxs']
        global_total_hc_idxs = asm_result['global_total_hc_idxs']
        writer.add_scalar('ASM/test_acc', best_prec1, global_step=best_epoch)

    else:
        meta_epochs = 10
        print(f'begin {meta_epochs} epochs asm_meta train')
        # return model is best_model
        best_model, best_prec1, best_epoch, global_total_uc_idxs, global_total_hc_idxs = \
            train_asm_epochs(model, meta_epochs, inner_meta_epochs=1, prefix='ASM')
        asm_result = {
            'best_prec1': best_prec1,
            'best_epoch': best_epoch,
            'global_total_uc_idxs': global_total_uc_idxs,
            'global_total_hc_idxs': global_total_hc_idxs
        }
        dump_json(asm_result, f'{exp_dir}/asm_result.json')

    print(f'Finish asm_meta train, best acc: {best_prec1}, best epoch: {best_epoch}')

    model = best_model

    ## fine-tune
    # lr = 1e-2  # todo: 太大，初始 acc 会降低
    # optimizer_s = torch.optim.SGD(model.params(), lr,
    #                               momentum=args.momentum, nesterov=args.nesterov,
    #                               weight_decay=args.weight_decay)

    # update label dataset
    label_dataset = CIFAR(
        data=np.append(label_dataset.data, unlabel_dataset.data[global_total_uc_idxs], axis=0),
        targets=np.append(label_dataset.targets, unlabel_dataset.targets[global_total_uc_idxs], axis=0),
        transform=transform_train
    )
    label_loader = DataLoader(label_dataset,
                              batch_size=args.batch_size,
                              drop_last=False,
                              shuffle=True, **kwargs)

    # hc data
    hc_data = unlabel_dataset.data[global_total_hc_idxs]
    hc_targets = unlabel_dataset.targets[global_total_hc_idxs]

    start_epoch = best_epoch + 1
    finetune_epochs = 20
    finetune_interval = 2  # 或许可以设置更大，因为 hc_acc 很高

    asm_dataloader = label_loader

    print(f'begin {finetune_epochs} epochs finetune train')

    for idx, epoch in enumerate(range(start_epoch, start_epoch + finetune_epochs)):
        # lr = lr * ((0.1 ** int(idx >= 10)) * (0.1 ** int(idx >= 15)))
        # set_lr(lr, optimizer_s)
        # writer.add_scalar('Finetune/lr', lr, global_step=epoch)

        if idx % finetune_interval == 0:
            # no shuffle
            hc_probs = detect_unlabel_imgs(model, hc_data, args.num_classes)
            hc_idxs, hc_preds = get_hc_samples(hc_probs, args.delta)
            hc_ratio = len(hc_idxs) / len(hc_probs)

            if len(hc_idxs) > 0:
                hc_gts = np.take(hc_targets, hc_idxs, axis=0)  # divide 0
                hc_acc = sum(hc_preds == hc_gts) / len(hc_gts)
            else:
                hc_acc = 0

            writer.add_scalar('Finetune/hc_ratio', hc_ratio, global_step=epoch)
            writer.add_scalar('Finetune/hc_acc', hc_acc, global_step=epoch)

            asm_dataset = CIFAR(
                data=np.append(label_dataset.data, hc_data[hc_idxs], axis=0),
                targets=np.append(label_dataset.targets, hc_preds, axis=0),
                transform=transform_train
            )
            asm_dataloader = DataLoader(asm_dataset,
                                        batch_size=args.batch_size,  # 保持和原模型相同 batchsize
                                        drop_last=False,
                                        shuffle=True, **kwargs)

        # train
        train_base(asm_dataloader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer, prefix='Finetune')

        best_model, best_prec1, best_epoch = valid_save_model(epoch, model, prefix='Finetune')

    print(f'Finish fine-tune train, best acc: {best_prec1}, best epoch: {best_epoch}')
