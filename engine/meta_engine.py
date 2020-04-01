import torch
import torch.nn.functional as F
from utils import AverageMeter, to_var
from engine.base_engine import accuracy


def train_meta(train_loader, model, criterion,
               optimizer_a,
               step, print_freq):  # step in total
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        output = model(input_var)

        # todo: weighted loss?
        loss = criterion(output, target_var)  # reduction='mean', [1,]
        prec_train = accuracy(output.data, target_var.data, topk=(1,))[0]

        # 普通更新
        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        losses.update(loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        # idx in trainloader
        if (i + 1) % print_freq == 0:
            print('Step:[{0}] [{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                step, i + 1, len(train_loader), loss=losses, top1=top1))

    return losses.avg, top1.avg


def train_meta_weight(train_loader, model, criterion,
                      optimizer_a,
                      step, print_freq):  # step in total
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        output = model(input_var)  # [100,10]

        loss_es = F.cross_entropy(output, target_var, reduction='none')  # [100,1]

        # use softmax to get sample weights
        weights = F.softmax(loss_es, dim=0)
        loss = torch.sum(weights * loss_es)

        # 普通更新
        optimizer_a.zero_grad()
        loss.backward()
        optimizer_a.step()

        prec_train = accuracy(output.data, target_var.data, topk=(1,))[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        # idx in trainloader
        if (i + 1) % print_freq == 0:
            print('Step:[{0}] [{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                step, i + 1, len(train_loader), loss=losses, top1=top1))

    return losses.avg, top1.avg
