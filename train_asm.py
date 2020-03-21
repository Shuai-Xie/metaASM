import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
import random
import numpy as np

from datasets.cls_datasets import CIFAR
from datasets.imb_data_utils import get_imb_meta_test_datasets, transform_train, transform_test
from net.resnet import ResNet32
from optim.adabound import AdaBound
from utils import get_curtime, save_model

from engine import adjust_lr, train_base, validate
from utils import detect_unlabel_imgs, get_select_fn, get_high_conf_samples

parser = argparse.ArgumentParser(description='Classification on cifar10')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
# parser.add_argument('--net', default='ResNet18', type=str,
#                     help='available models in net/classifier dir')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',  # train bs
                    help='input batch size for training (default: 100)')
parser.add_argument('--num_classes', type=int, default=10)  # cifar10
parser.add_argument('--num_meta', type=int, default=10,  # meta data, 10 samples/class
                    help='The number of meta data for each class.')
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
parser.add_argument('-ce', '--cost_effective', default=True,
                    help="whether to use Cost Effective high confidence sample pseudo-labeling. default: True")
parser.add_argument('--split', default=0.1, type=float,  # 5000 samples
                    help='Initial Annotated Samples Ratio. default: 0.1')
parser.add_argument('--ratio', default=1.0, type=float,  # 5000 samples
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
    '--num_meta', '0',
    '-K', '2000',
    '--init_epochs', '10',  # 试试更多 epoch
    '--split', '0.4',
    '--ratio', '1',  # 小样本
    '--tag', 'asm_adam'
]
args = parser.parse_args(params)
pprint(vars(args))

cudnn.benchmark = True

# get datasets
random.seed(args.seed)
label_dataset, unlabel_dataset, train_meta_dataset, test_dataset = get_imb_meta_test_datasets(
    args.dataset, args.num_classes, args.num_meta, args.imb_factor, args.split, args.ratio
)  # CIFAR10

# imb_train/valid_meta/test
kwargs = {'num_workers': 4, 'pin_memory': True}

# 将 label/unlabel loader 放入同一 batch
label_loader = DataLoader(label_dataset,
                          batch_size=args.batch_size,
                          drop_last=False,
                          shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)

"""steps
1. train init_epochs on labelset
2. infer on unlabelset [use all]
    2.1 select all hc samples by delta (entropy threshold)
    2.2 select topK uc samples by uncertain_criterion, if mixed with hc, use hc
3. update label_loader
4. retrain
5. goto 2
"""

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
    # 60 epoch 后可能需要降低 lr, acc 不像 sgd 震荡较大
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

    # pre-define hc
    hc_data = np.empty([0] + list(label_dataset.data.shape[1:]), dtype='uint8')  # (0,32,32,3)
    hc_preds = np.empty([0], dtype='int64')
    hc_idxs, uc_idxs = [], []
    total_uc_num = 0  # asm 总共使用的 uc 样本数

    use_asm = True
    for epoch in range(1, args.epochs + 1):
        # 调整 classifier optimizer 的 lr = meta_lr
        adjust_lr(args.lr, optimizer_a, epoch, writer)

        # train on (imb_train_data)
        writer.add_scalar('ASM/batches', len(label_loader), global_step=epoch)

        train_base(label_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion,
                         epoch, args.print_freq, writer)

        # remember best prec@1 and save checkpoint
        if prec1 > best_prec1:
            best_prec1, best_epoch = prec1, epoch
            save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(epoch)),
                       model, epoch, best_prec1)
            # 每次 prec 提升后再用 asm 是个好方法，但是太理想
            # 我们恰恰希望能引入 active samples 来带来 prec 提升

        # begin asm stage
        if use_asm and epoch >= args.init_epochs and len(unlabel_dataset) > args.uncertain_samples_size:
            # 1.detect on unlabel images
            y_pred_prob = detect_unlabel_imgs(model, unlabel_dataset.data,  # on total unlabel pool
                                              args.num_classes, args.batch_size)
            # 2.split label/unlabel dataset
            # 以 hc 为主，舍弃 uc 中相同的，如果先选 uc， K 就定量了

            # 2.1 add hc samples to label_loader, not add to label_dataset(only labeled)
            # Self Learning [machine label]
            if args.cost_effective:
                hc_idxs, hc_preds = get_high_conf_samples(y_pred_prob, args.delta)  # entropy thre, note num_classes
                hc_ratio = len(hc_idxs) / len(unlabel_dataset)
                writer.add_scalar('ASM/hc_ratio', hc_ratio, global_step=epoch)

                if hc_ratio > 0.9:  # hc 占比 unlabel 很高时，unlabel 可用信息正在减少
                    use_asm = False  # note! not use asm here!
                    # 加上这批 hc, label_loader 不再更新

                if len(hc_idxs) > 0:  # 存在 hc 样本
                    hc_data = np.take(unlabel_dataset.data, hc_idxs, axis=0)
                    # note: hc_acc 实际上不可获得
                    hc_gts = np.take(unlabel_dataset.targets, hc_idxs, axis=0)
                    hc_acc = sum(hc_preds == hc_gts) / len(hc_gts)
                    writer.add_scalar('ASM/hc_acc', hc_acc, global_step=epoch)

            # 2.2 add 'ann' uc_idxs to label_dataset
            # Active Learning [human label]
            if use_asm:
                _, uc_idxs = select_fn(y_pred_prob, args.uncertain_samples_size)  # top 1000, np array 不会越界

                # rm samples already selected by hc
                uc_idxs = [i for i in uc_idxs if i not in hc_idxs]
                total_uc_num += len(uc_idxs)

                # add/rm uc samples to label/unlabel dataset
                label_dataset.data = np.append(label_dataset.data,  # append by data
                                               np.take(unlabel_dataset.data, uc_idxs, axis=0), axis=0)
                label_dataset.targets = np.append(label_dataset.targets,
                                                  np.take(unlabel_dataset.targets, uc_idxs, axis=0), axis=0)
                unlabel_dataset.data = np.delete(unlabel_dataset.data, uc_idxs, axis=0)  # rm by idxs
                unlabel_dataset.targets = np.delete(unlabel_dataset.targets, uc_idxs, axis=0)

            # 3.update label_loader for next epoch train
            # 不能直接更新原来 label_loader.dataset
            # ValueError: dataset attribute should not be set after DataLoader is initialized
            asm_dataset = CIFAR(  # ori_label + AL(top_uncer) + SL(hc)
                data=np.append(label_dataset.data, hc_data, axis=0),
                targets=np.append(label_dataset.targets, hc_preds, axis=0),
                transform=transform_train,
            )
            label_loader = DataLoader(asm_dataset,
                                      batch_size=args.batch_size,
                                      drop_last=False,
                                      shuffle=True, **kwargs)
        # record samples num
        writer.add_scalars('ASM/samples', {
            # train = hc + label
            # total = label + unlabel
            'train': len(label_loader.dataset),
            'hc': len(hc_idxs),
            'uc': total_uc_num,
            'label': len(label_dataset),
            'unlabel': len(unlabel_dataset)
        }, global_step=epoch)

    print('Best accuracy: {}, epoch: {}'.format(best_prec1, best_epoch))
