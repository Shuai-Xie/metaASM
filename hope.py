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

from datasets import *
from datasets.build_imb_dataset import get_imb_meta_test_datasets

from net import *
from engine import *
from utils import *
from optim import *

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
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,  # decay
                    help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default=42, metavar='S',  # random
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--tag', default='exp', type=str,
                    help='experiment tag to create tensorboard, model_save_dir, exp_dir')
# asm
parser.add_argument('--ckpt', type=str,
                    help='already pretrained model')
parser.add_argument('--ckpt_asm', type=str,
                    help='already pretrained model')
parser.add_argument('--split', default=0.1, type=float,
                    help='Initial Annotated Samples Ratio. default: 0.1')
parser.add_argument('--ratio', default=1.0, type=float,
                    help='use how many samples of each cls')
parser.add_argument('--uc_batchsize', default=1000, type=int,
                    help='Uncertain samples selection size. default: 1000')
parser.add_argument('--max_uc_ratio', default=0.5, type=float,
                    help='Uncertain samples selection size. default: 1000')
parser.add_argument('-uc', '--uncertain_criterion', default='ms', type=str,
                    help='Uncertain selection Criteria:\n'
                         'rs(Random Sampling)\n'
                         'lc(Least Confidence)\n'
                         'ms(Margin Sampling)\n'
                         'en(Entropy)')
# 论文中让 delta decay，理由是学习后期 hc_sample 更多，可能引入更多 incorrect auto ann；
# 实际 cifar10 影响不大；控制其不变起到引入更多样本作用
parser.add_argument('--hc_delta', default=0.5, type=float,  # prob > 0.9, 越大 hc_acc 越低
                    help="High confidence samples selection threshold")
parser.add_argument('--uc_delta', default=1.0, type=float,
                    help="Uncertain samples selection threshold")
parser.add_argument('--patience', default=5, type=int,
                    help='stop training if test_acc does not increase in patience epochs')
parser.add_argument('--note', type=str, help="train configs note")

params = [
    '--dataset', 'cifar10',
    '--num_classes', '10',
    '--imb_factor', '1',
    # important params
    # '--num_meta', '10',  # 从 labelset 选出的代表性 sample 数量，恰好 = bs?
    '-uc', 'ms',
    '--uc_batchsize', '3000',  # 10
    '--max_uc_ratio', '0.5',
    '--split', '0.4',
    '--ratio', '1',  # 小样本
    # '--ckpt', 'output/padam4_cifar10_imb1_s0.4_r1.0_m300_Apr03_114830/init_model.pth',
    # '--ckpt_asm', 'output/padam4_cifar10_imb1_s0.4_r1.0_m300_Apr02_224414/asm_model.pth',
    # '--ckpt_asm', 'output/padam8_cifar10_imb1_s0.4_r1.0_m300_Apr03_114956/asm_model.pth',
    '--tag', 'padam4_wd'
]
args = parser.parse_args(params)
args.uncertain_samples_size = int(args.uc_batchsize * args.max_uc_ratio)
args.num_meta = int(args.uc_batchsize / args.num_classes)  # 数据越多 meta_epoch 增长越快
args.note = """
1.设置 patience，在 test_acc 没增加后，自动停下训练; 不再设置每个阶段的 num_epochs
2.unlabel dataset，每个 B_U 只过一遍
3.调大 uc_batchsize = 3000
4.asm_batch patience=3
"""

pprint(vars(args))

# cudnn.benchmark = True

# get datasets
random.seed(args.seed)  # 保证初始划分 label/unlabel 一致
label_dataset, unlabel_dataset, _, test_dataset = get_imb_meta_test_datasets(
    args.dataset, args.num_classes, 0, args.imb_factor, args.split, args.ratio
)  # CIFAR10
random.seed()  # 解除

# imb_train/valid_meta/test
kwargs = {'num_workers': 4, 'pin_memory': True}  # pin 训练快

# 将 label/unlabel loader 放入同一 batch
label_loader = DataLoader(label_dataset,
                          batch_size=args.batch_size,
                          shuffle=True, **kwargs)
unlabel_loader = DataLoader(unlabel_dataset,  # DataLoader will cvt np to tensor
                            batch_size=args.uc_batchsize,
                            shuffle=True, **kwargs)  # shuffle 能带来更稳定的 imb, 人工设置保证重现
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)

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

# global best model
best_model, best_acc, best_epoch = None, 0, 0

acc_caches = AccCaches(patience=args.patience)


def valid_save_model(model, epoch, acc_caches, optimizer, prefix, ckpt):
    global best_model, best_acc, best_epoch

    # valid
    _, acc = validate(test_loader, model, criterion,
                      epoch, args.print_freq, writer, prefix)

    acc_rise = False  # flag to re infer on hc
    if acc > best_acc:
        best_model, best_acc, best_epoch = model, acc, epoch
        save_model(f'{model_save_dir}/{ckpt}', best_model, best_epoch, best_acc, optimizer)  # 覆盖保存
        print(f'{prefix} epoch {epoch}, best acc: {best_acc}, best epoch: {best_epoch}')
        acc_rise = True

    acc_caches.add(epoch, acc)

    # 实际启动条件：
    # best_acc 之后 patience epochs 内 best_acc 无上升
    # 而不是每次都从 patience 中选取最优 epoch 作为下次训练 model
    stop_train = False
    if acc_caches.full():
        print('acc caches:', acc_caches.accs)
        print('best acc:', best_epoch, best_acc)
        _, max_acc = acc_caches.max_cache_acc()
        if max_acc < best_acc:  # stop train
            stop_train = True

    return stop_train, acc_rise


def train_init_epochs(model, prefix='Init', ckpt='init_model.pth'):
    epoch = best_epoch
    acc_caches.reset()

    while True:
        epoch += 1
        # train
        train_base(label_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer, prefix)
        # valid
        stop_train, _ = valid_save_model(model, epoch, acc_caches, optimizer_a, prefix, ckpt)

        if stop_train:
            break


def get_asm_dataloader(model, epoch, prefix='Finetune'):
    hc_data, hc_targets = unlabel_dataset.data, unlabel_dataset.targets

    hc_probs = detect_unlabel_imgs(model, hc_data, args.num_classes)
    hc_idxs, hc_preds = get_hc_samples(hc_probs, args.hc_delta)
    hc_ratio = len(hc_idxs) / len(hc_probs)

    if len(hc_idxs) > 0:
        hc_gts = np.take(hc_targets, hc_idxs, axis=0)  # divide 0
        hc_acc = sum(hc_preds == hc_gts) / len(hc_gts)
    else:
        hc_acc = 0

    writer.add_scalar(f'{prefix}/hc_ratio', hc_ratio, global_step=epoch)
    writer.add_scalar(f'{prefix}/hc_acc', hc_acc, global_step=epoch)

    asm_dataset = CIFAR(
        data=np.vstack((label_dataset.data, hc_data[hc_idxs])),
        targets=np.hstack((label_dataset.targets, hc_preds)),
        transform=transform_train
    )
    asm_dataloader = DataLoader(asm_dataset,
                                batch_size=args.batch_size,  # 保持和原模型相同 batchsize
                                shuffle=True, **kwargs)

    return asm_dataloader


def train_finetune_epochs(model, hc_interval=5, prefix='Finetune', ckpt='finetune_model.pth'):
    epoch = best_epoch
    acc_caches = AccCaches(patience=5)  #
    acc_caches.reset()

    # asm_dataloader = get_asm_dataloader(model, epoch)

    # only label
    asm_dataloader = DataLoader(label_dataset,
                                batch_size=args.batch_size,
                                shuffle=True, **kwargs)

    # total
    # asm_dataset = CIFAR(
    #     data=np.vstack((label_dataset.data, unlabel_dataset.data)),
    #     targets=np.hstack((label_dataset.targets, unlabel_dataset.targets)),
    #     transform=transform_train
    # )
    # asm_dataloader = DataLoader(asm_dataset,
    #                             batch_size=args.batch_size,
    #                             shuffle=True, **kwargs)

    # finetune_lr = 1e-2
    # optimizer_a = torch.optim.SGD(model.params(), finetune_lr,
    #                               momentum=args.momentum, nesterov=True,
    #                               weight_decay=args.weight_decay)

    # finetune_lr = 3e-4
    # optimizer_a = AdamW(model.params(),
    #                     lr=finetune_lr,
    #                     betas=(0.9, 0.999), eps=1e-8, amsgrad=True)

    # optimizer_a = SWATS(model.params(),
    #                     lr=1e-3,
    #                     betas=(0.9, 0.999), eps=1e-8, amsgrad=True,
    #                     weight_decay=args.weight_decay, nesterov=args.nesterov)

    while True:
        epoch += 1
        torch.manual_seed(random.randint(0, 10000))
        cur_lr = get_lr(optimizer_a)
        print('current lr:', cur_lr)
        writer.add_scalar(f'{prefix}/lr', cur_lr, global_step=epoch)

        # train
        train_base(asm_dataloader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer, prefix)
        # valid
        stop_train, acc_rise = valid_save_model(model, epoch, acc_caches, optimizer_a, prefix, ckpt)

        # if acc_rise:
        #     asm_dataloader = get_asm_dataloader(model, epoch)

        # if stop_train:
        #     break

        if stop_train:  # 停下训练，自动减小 lr，默认一定有 finetune_model.pth
            # model, optimizer_a, _, _ = load_model(model, f'{model_save_dir}/{ckpt}', optimizer_a)
            cur_lr /= 10
            if cur_lr < 1e-4:
                break
            set_lr(cur_lr, optimizer_a)
            acc_caches.reset()


def train_asm_epochs(label_dataset, unlabel_dataset):
    global best_model, best_acc, best_epoch

    epoch = best_epoch + 1
    ab_acc_caches = AccCaches(patience=3)  # ab: asm_batch
    global_uc_idxs = []

    torch.manual_seed(args.seed)  # 控制 B_U 相同，且均衡

    for i, (input, target, img_idxs) in enumerate(unlabel_loader):
        ab_epoch = 0
        ab_acc_caches.reset()

        model = best_model
        best_ab_epoch = -1

        # img_idxs of each cls
        # cls_img_idxs_dict = get_cls_img_idxs_dict(label_dataset.targets, args.num_classes)
        # sort_cls_idxs_dict = sort_cls_samples(model, label_dataset, args.num_classes,
        #                                       criterion=args.uncertain_criterion)

        # 从 labelset 随机抽取 与 B_U 等量级的 D_M
        # meta_dataset = random_sample_equal_cls(label_dataset, cls_img_idxs_dict, args.num_meta)
        # meta_dataset = random_system_sample_meta_dataset(label_dataset, sort_cls_idxs_dict, args.num_meta)

        # 使用全部 labelset
        meta_dataset = label_dataset

        prefix = f'ASM_{i}'  # B_U idxs
        print(f'begin {prefix} train')

        # cvt tensor to np
        input_np, target_np = input.numpy().astype('uint8'), target.numpy().astype('int64')
        img_idxs = img_idxs.numpy().astype('int64')

        batch_all_uc_idxs = np.array([], dtype=np.int)  # record total uc_samples in a B_U，慎用 empty

        while True:
            ab_epoch += 1

            ## split & record asm split
            # results = asm_split_hc_uc_delta(model, input, target,
            #                                 args.hc_delta,
            #                                 args.uc_delta)
            results = asm_split_hc_delta_uc_K(model, input_np, target_np,  # 可保证最终未标注拥有更高准确率
                                              args.hc_delta,
                                              uc_select_fn, args.uncertain_samples_size)
            # input 顺序固定，可在 while 之外映射到 global uc_idxs

            # uc
            batch_cur_uc_idxs = results['uc']['idxs']  # idxs in input range
            batch_all_uc_idxs = np.union1d(batch_all_uc_idxs, batch_cur_uc_idxs)  # gts

            # B_U 只采用当前 ab_epoch 划分结果
            # hc:uc = 1:2
            # hc_data, hc_targets = results['hc']['data'], results['hc']['targets']
            # uc_data, uc_targets = results['uc']['data'], results['uc']['targets']
            # uc_data, uc_targets = multi_uc_data(uc_data, uc_targets, factor=2)

            # B_U 采用之前 ab_epoch 选出的总共 uc_samples，因为已经标注了，自然也作为 gt 使用
            uc_data, uc_targets = input_np[batch_all_uc_idxs], target_np[batch_all_uc_idxs]  # gts
            # uc_data, uc_targets = multi_uc_data(uc_data,uc_targets,factor=2)

            # hc
            # 如果 hc_idx 已在 all_uc_idxs 中出现，过滤掉，使用 gt
            hc_idxs, hc_data, hc_targets = results['hc']['idxs'], results['hc']['data'], results['hc']['targets']
            keep_idxs = [k for k in range(len(hc_idxs)) if hc_idxs[k] not in batch_all_uc_idxs]  # idx k in hc range
            hc_data, hc_targets = hc_data[keep_idxs], hc_targets[keep_idxs]  # preds

            asm_meta_dataset = CIFAR(
                data=np.vstack((meta_dataset.data, uc_data, hc_data)),
                targets=np.hstack((meta_dataset.targets, uc_targets, hc_targets)),
                transform=transform_train
            )
            torch.manual_seed(random.randint(0, 10000))
            asm_meta_loader = DataLoader(asm_meta_dataset,
                                         batch_size=args.batch_size,  # 保持和原模型相同 batchsize
                                         shuffle=True, **kwargs)

            # train, note: 不要用错 dataloader
            train_loss, train_acc = train_base(asm_meta_loader, model, criterion,
                                               optimizer_a,
                                               epoch, args.print_freq)  # not record to tensorboard
            # valid
            _, acc = validate(test_loader, model, criterion,
                              epoch, args.print_freq)

            if acc > best_acc:
                best_model, best_acc, best_epoch, best_ab_epoch = model, acc, epoch, ab_epoch
                save_model(f'{model_save_dir}/asm_model.pth', best_model, best_epoch, best_acc)  # 覆盖保存
                print(f'{prefix}, best acc: {best_acc}, best ab_epoch: {best_ab_epoch}')

            ab_acc_caches.add(ab_epoch, acc)

            if ab_acc_caches.full():  # 从 >= patience 开始记录
                print('ab acc caches:', ab_acc_caches.accs)
                print(f'{prefix} best acc:', best_ab_epoch, best_acc)  # 如果没有提升 就是 -1
                _, max_acc = ab_acc_caches.max_cache_acc()
                if max_acc < best_acc:
                    print(f'Finish {prefix} train, best acc: {best_acc}, best ab_epoch: {best_ab_epoch}')
                    break

        batch_all_uc_idxs = img_idxs[batch_all_uc_idxs]  # cvt to global

        # update labelset
        label_dataset = CIFAR(
            data=np.vstack((label_dataset.data, unlabel_dataset.data[batch_all_uc_idxs])),
            targets=np.hstack((label_dataset.targets, unlabel_dataset.targets[batch_all_uc_idxs])),
            transform=transform_train
        )
        writer.add_scalars('ASM/dataset', {
            'label': len(label_dataset),
            'unlabel': 50000 - len(label_dataset),
        }, global_step=i)
        writer.add_scalars('ASM/acc', {
            'train': train_acc,
            'test': best_acc
        }, global_step=i)
        writer.add_scalar('ASM/train_loss', train_loss, global_step=i)  # i: B_U idx
        writer.add_scalar('ASM/batch_epochs', ab_epoch, global_step=i)  # last ab_epoch is the num of batch_epochs
        writer.add_scalar('ASM/batch_uc_sample', len(batch_all_uc_idxs), global_step=i)

        global_uc_idxs.extend(batch_all_uc_idxs.tolist())

    # update final unlabel dataset, hc
    unlabel_dataset = CIFAR_unlabel(
        data=np.delete(unlabel_dataset.data, global_uc_idxs, axis=0),
        targets=np.delete(unlabel_dataset.targets, global_uc_idxs, axis=0),
        transform=None
    )

    # save for retrain
    asm_result = {
        'labelset': (label_dataset.data, label_dataset.targets),  # pickle can't save lambda
        'unlabelset': (unlabel_dataset.data, unlabel_dataset.targets),
    }
    dump_pickle(asm_result, out_path=f'{exp_dir}/asm_result.pkl')

    return label_dataset, unlabel_dataset


if __name__ == '__main__':
    # build model
    model = ResNet32(args.num_classes).cuda()
    print('build model done!')

    # Adam
    # optimizer_a = torch.optim.Adam(model.params(),
    #                                lr=1e-3,
    #                                betas=(0.9, 0.999), eps=1e-8, amsgrad=True)  # 加 ams 训练略慢

    # optimizer_a = AdamW(model.params(),
    #                     lr=1e-3,
    #                     betas=(0.9, 0.999), eps=1e-8, amsgrad=True)

    optimizer_a = Padam(model.params(),
                        lr=1e-1,
                        betas=(0.9, 0.999), eps=1e-8,
                        weight_decay=args.weight_decay,
                        partial=1 / 4)

    criterion = nn.CrossEntropyLoss().cuda()
    uc_select_fn = get_select_fn(args.uncertain_criterion)

    # init train

    if args.ckpt is not None:
        print('load init_trained model')
        best_model, optimizer_a, best_acc, best_epoch = load_model(model, args.ckpt, optimizer_a)
        writer.add_scalar('Init/test_acc', best_acc, global_step=best_epoch)
    else:
        if args.ckpt_asm is None:
            print('begin init train')
            train_init_epochs(model, prefix='Init', ckpt='init_model.pth')  # best save as global

    print(f'Finish init train, best acc: {best_acc}, best epoch: {best_epoch}')

    # asm train
    if args.ckpt_asm is not None:
        print('load asm model')
        best_model, optimizer_a, best_acc, best_epoch = load_model(model, args.ckpt_asm, optimizer_a)
        writer.add_scalar('ASM/test_acc', best_acc, global_step=best_epoch)
        # load batch asm split whole dataset
        asm_result = load_pickle(f'exp/{args.ckpt_asm.split("/")[1]}/asm_result.pkl')
        label_dataset = CIFAR(asm_result['labelset'][0], asm_result['labelset'][1], transform_train)
        unlabel_dataset = CIFAR_unlabel(asm_result['unlabelset'][0], asm_result['unlabelset'][1])
    else:
        print('begin asm train')  # split uc/hc in batch
        label_dataset, unlabel_dataset = train_asm_epochs(label_dataset, unlabel_dataset)

    print(f'Finish asm train, best acc: {best_acc}, best epoch: {best_epoch}')

    # finetune train
    print('begin finetune train')
    train_finetune_epochs(best_model, prefix='Finetune', ckpt='finetune_model.pth')
    print(f'Finish finetune train, best acc: {best_acc}, best epoch: {best_epoch}')
