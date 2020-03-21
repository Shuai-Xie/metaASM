import torch
from torch.autograd import Variable
import time
import numpy as np


class AverageMeter(object):
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


def cmp_two_list(l1, l2):
    """
    return True if two list 不存在相同元素
    """
    assert len(l1) == len(l2)
    for e in l1:
        if e in l2:
            return False
    return True


def lasso_shift(records):
    mean = np.mean(records)
    return np.mean([abs(x - mean) for x in records])


def ridge_shift(records):
    mean = np.mean(records)  # = np.var(records)
    return np.mean([(x - mean) ** 2 for x in records])


def to_var(x, requires_grad=True):
    # .cuda(), torch.autograd.Variable
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time
