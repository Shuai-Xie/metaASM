import torch
from torch.autograd import Variable
from datetime import datetime
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


def lasso_shift(records):
    mean = np.mean(records)
    return np.mean([abs(x - mean) for x in records])


def ridge_shift(records):
    mean = np.mean(records)  # = np.var(records)
    return np.mean([(x - mean) ** 2 for x in records])


def to_var(x, requires_grad=True):
    # 转成 torch.autograd.Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()


def get_curtime():
    current_time = datetime.now().strftime('%b%d_%H%M%S')
    return current_time


def load_model(model, ckpt_path, optimizer=None):
    """
    model 和 optimizer 从 ckpt_path load_state_dict
    """
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    print('load {}, epoch {}'.format(ckpt_path, ckpt['epoch']))

    best_epoch = ckpt.get('epoch', -1)
    best_acc = ckpt.get('accuracy', 0)

    if optimizer is not None:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        return model, optimizer, best_epoch, best_acc
    else:
        return model


def save_model(ckpt_path, model, epoch, accuracy, optimizer=None):
    # model = { 'epoch': , 'state_dict': , 'optimizer' }
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()  # convert data_parallal to model
    else:
        state_dict = model.state_dict()
    ckpt = {
        'epoch': epoch,
        'accuracy': accuracy,
        'state_dict': state_dict
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    torch.save(ckpt, ckpt_path)
    print('save {}, epoch {}'.format(ckpt_path, ckpt['epoch']))
