import torch
import torch.nn.functional as F
from utils import AverageMeter, to_var
from engine.base_train import accuracy
import copy


def train_with_vnet(train_loader, valid_loader,
                    model, vnet,
                    lr,  # meta_model use same lr as model
                    optimizer_a, optimizer_c,
                    epoch, print_freq, writer):
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()

    begin_step = epoch * len(train_loader)
    meta_update_step = 10

    model.train()

    # todo: 验证 meta_dataset 是否需要更新
    # train_loader 是否需要每个 epoch 后更新
    # 不更新，最后统一计算 uc samples，code 简单

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        if i % meta_update_step == 0:
            # train on meta_model
            meta_model = copy.deepcopy(model)
            y_f_hat = meta_model(input_var)  # pred, hat

            cost_w = F.cross_entropy(y_f_hat, target_var, reduction='none')  # [100,1]
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))

            v_lambda = vnet(cost_v)  # [100, 1] sigmoid_()
            v_sum = torch.sum(v_lambda)  # 100 个 weights 之和
            v_lambda_norm = v_lambda / v_sum if v_sum != 0 else v_lambda

            l_f_meta = torch.sum(cost_v * v_lambda_norm)

            meta_model.zero_grad()
            grads = torch.autograd.grad(  # return backward grads
                l_f_meta,  # outputs
                (meta_model.params()),  # inputs, graph leaves
                create_graph=True,
                only_inputs=True,
            )
            meta_model.update_params(lr_inner=lr, source_params=grads)
            del grads
            writer.add_scalar('Train/meta_lr', lr, global_step=epoch)

            # update vnet
            val_input, val_target = next(iter(valid_loader))
            val_input_var = to_var(val_input, requires_grad=False)
            val_target_var = to_var(val_target, requires_grad=False)

            y_g_hat = meta_model(val_input_var)
            l_g_meta = F.cross_entropy(y_g_hat, val_target_var)

            prec_meta = accuracy(y_g_hat.data, val_target_var.data, topk=(1,))[0]

            optimizer_c.zero_grad()
            l_g_meta.backward()  # 反向传播，vnet 更新
            optimizer_c.step()

            meta_losses.update(l_g_meta.item(), n=1)
            meta_top1.update(prec_meta.item(), input.size(0))

        # Updating parameters of classifier network
        y_f = model(input_var)  # [100,10]
        cost_w = F.cross_entropy(y_f, target_var, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))  # [100,1] bs=100 for vnet

        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        # vnet 更新后，计算 new weight
        with torch.no_grad():
            w_new = vnet(cost_v)

        v_sum = torch.sum(w_new)
        w_v = w_new / v_sum if v_sum != 0 else w_new

        # 优化 outer model
        l_f = torch.sum(cost_v * w_v)
        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()  # 更新 model 参数

        # 更新 vnet 后，在 trian data 上的 weighted CE loss[计入新的weighted] 和 acc
        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        writer.add_scalars('Train/loss', {
            'meta_loss': meta_losses.avg,
            'train_loss': losses.avg
        }, global_step=begin_step + i)

        writer.add_scalars('Train/top1_acc', {
            'meta_acc': meta_top1.avg,
            'train_acc': top1.avg
        }, global_step=begin_step + i)

        # idx in trainloader
        if (i + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'  # current val, avg
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i + 1, len(train_loader),
                loss=losses, meta_loss=meta_losses, top1=top1, meta_top1=meta_top1))
