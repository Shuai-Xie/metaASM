import torch
from torch.autograd import Variable
import time
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccCaches:
    def __init__(self, patience):
        self.accs = []  # [(epoch, acc), ...]
        self.patience = patience

    def reset(self):
        self.accs = []

    def add(self, epoch, acc):
        if len(self.accs) >= self.patience:  # 先满足 =
            self.accs = self.accs[1:]  # 队头出队列
        self.accs.append((epoch, acc))  # 队尾添加

    def full(self):
        return len(self.accs) == self.patience

    def max_cache_acc(self):
        max_id = int(np.argmax([t[1] for t in self.accs]))  # t[1]=acc
        max_epoch, max_acc = self.accs[max_id]
        return max_epoch, max_acc


def cvt_iter_to_list(iter, type):
    return [type(v) for v in iter]


def cmp_two_list(l1, l2):
    """
    return True if two list 不存在相同元素
    """
    assert len(l1) == len(l2)
    for e in l1:
        if e in l2:
            return False
    return True


def multi_array(a, factor=3):
    if len(a.shape) == 1:
        return np.hstack([a] * factor)
    else:
        return np.vstack([a] * factor)


def multi_uc_data(uc_data, uc_targets, factor):
    uc_data = multi_array(uc_data, factor)
    uc_targets = multi_array(uc_targets, factor)
    return uc_data, uc_targets


def lasso_shift(records):
    mean = np.mean(records)
    return np.mean([abs(x - mean) for x in records])


def ridge_shift(records):
    mean = np.mean(records)  # = np.var(records)
    return np.mean([(x - mean) ** 2 for x in records])


def to_var(x, requires_grad):
    # .cuda(), torch.autograd.Variable
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()


def empty_x(img_shape):
    return np.empty([0] + list(img_shape), dtype='uint8')  # (0,32,32,3)


def empty_y():
    return np.empty([0], dtype='int64')


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time
