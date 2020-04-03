import torch
from utils import AverageMeter, to_var


def update_lr(lr, epoch):
    # lr = lr * (0.1 ** int(epoch >= 40))
    lr = lr * ((0.1 ** int(epoch >= 30)) * (0.1 ** int(epoch >= 40)))
    # lr = lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 45)))
    # lr = lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    # lr = lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 70)))
    return lr


def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 可直接修改 optimizer 参数；可修改 partial


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']  # list[dict{'lr'}]


def adjust_lr(lr, optimizer, epoch, writer):
    # divided by 10 after 80 and 90 epoch (for a total 100 epochs)
    # 这种写法很简洁
    lr = update_lr(lr, epoch)
    set_lr(lr, optimizer)
    writer.add_scalar('Train/lr', lr, global_step=epoch)
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))  # acc%
    return res


@torch.no_grad()
def validate(valid_loader, model, criterion,
             epoch, print_freq, writer=None, prefix='Test'):
    """Perform validation on the validation set"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (input, target) in enumerate(valid_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var.data, topk=(1,))[0]

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time

        if (i + 1) % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i + 1, len(valid_loader), loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    if writer:
        writer.add_scalar(f'{prefix}/test_loss', losses.avg, global_step=epoch)
        writer.add_scalar(f'{prefix}/test_acc', top1.avg, global_step=epoch)

    return losses.avg, top1.avg


def train_base(train_loader, model, criterion,
               optimizer_a,
               epoch, print_freq, writer=None, prefix='Train'):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    begin_step = epoch * len(train_loader)

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        output = model(input_var)
        loss = criterion(output, target_var)  # reduction='mean', [1,]
        prec_train = accuracy(output.data, target_var.data, topk=(1,))[0]

        # 普通更新
        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        # CE loss, reduction='mean'
        losses.update(loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        if writer:
            writer.add_scalar(f'{prefix}/train_loss', losses.avg, global_step=begin_step + i)

        # idx in trainloader
        if print_freq:
            if (i + 1) % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i + 1, len(train_loader), loss=losses, top1=top1))

    if writer:
        writer.add_scalar(f'{prefix}/train_loss_epoch', losses.avg, global_step=epoch)
        writer.add_scalar(f'{prefix}/train_acc', top1.avg, global_step=epoch)

    return losses.avg, top1.avg
