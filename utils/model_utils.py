import torch
from utils.base_utils import get_list_mul


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
        return model, optimizer, best_acc, best_epoch
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
    torch.save(ckpt, ckpt_path)  # 可以覆盖保存
    print('save {}, epoch {}'.format(ckpt_path, ckpt['epoch']))


def print_model_named_params(model):
    """
    model.named_parameters(): 生成 (name, param)
        yielding both the name of the parameter as well as the parameter itself
        print, :< 左对齐, :> 右对齐; 默认情况 str 左对齐，number 右对齐
    """
    print('=> model named params:')
    print('{:4} {:50} {:30} {:10} {}'.format('idx', 'name', 'size', 'params', 'grad'))
    for idx, (name, param) in enumerate(model.named_parameters()):  # generator, 生成 name, param tensor
        print('{:<4} {:50} {:30} {:<10} {}'.format(
            idx, name, str(param.size()), get_list_mul(param.size()), param.requires_grad))
