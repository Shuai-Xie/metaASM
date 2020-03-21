import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
from datasets.imb_data_utils import get_imb_meta_test_datasets
from net.resnet import ResNet32
from pprint import pprint
from utils import get_curtime, save_model
from engine import *

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

params = [
    '--dataset', 'cifar10',
    '--num_classes', '10',
    '--imb_factor', '1',
    '--num_meta', '0',
    '--tag', 'base'
]
args = parser.parse_args(params)
pprint(vars(args))

cudnn.benchmark = True

# get datasets
random.seed(args.seed)
train_dataset, _, test_dataset = get_imb_meta_test_datasets(
    args.dataset, args.num_classes, args.num_meta, args.imb_factor
)

# imb_train/valid_meta/test
kwargs = {'num_workers': 4, 'pin_memory': True}
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          drop_last=False,
                          shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)

if __name__ == '__main__':
    exp = f'{args.tag}_{args.dataset}_imb{args.imb_factor}_{get_curtime()}'
    print('exp:', exp)

    model = ResNet32(args.num_classes).cuda()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    print('build model done!')

    criterion = nn.CrossEntropyLoss().cuda()  # 内部实现 F.cross_entropy

    writer = SummaryWriter(log_dir=os.path.join('runs', exp))
    model_save_dir = os.path.join('output', exp)
    os.makedirs(model_save_dir, exist_ok=True)

    best_prec1, best_epoch = 0, 0

    for epoch in range(1, args.epochs + 1):
        # 调整 classifier optimizer 的 lr = meta_lr
        adjust_lr(args.lr, optimizer_a, epoch, writer)

        # train on (imb_train_data)
        train_base(train_loader, model, criterion,
                   optimizer_a,
                   epoch, args.print_freq, writer)

        # evaluate on validation set
        _, prec1 = evaluate(test_loader, model, criterion,
                            epoch, args.print_freq, writer)

        # remember best prec@1 and save checkpoint
        if prec1 > best_prec1:
            best_prec1, best_epoch = prec1, epoch
            save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(epoch)),
                       model, epoch, best_prec1)

    print('Best accuracy: {}, epoch: {}'.format(best_prec1, best_epoch))
