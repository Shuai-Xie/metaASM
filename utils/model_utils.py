import torch


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
    torch.save(ckpt, ckpt_path)
    print('save {}, epoch {}'.format(ckpt_path, ckpt['epoch']))
