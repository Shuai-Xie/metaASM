from net import ResNet32
from utils import load_model, get_list_mul
import torch


def print_model_params(model):
    """
    model.parameters(): typically passed to an optimizer. 与下面相比，只少了 name
        param type: torch.nn.parameter.Parameter, Parameter(torch.Tensor)
    """
    print('=> model params:')
    k = 0
    print('{:4} {:30} {:10} {}'.format('idx', 'size', 'params', 'grad'))
    for idx, param in enumerate(model.params()):  # generator
        print('{:<4} {:30} '.format(idx, str(param.size())), end='')
        multi = get_list_mul(param.size())  # [out_C, in_C, kernel_w, kernel_h]
        print('{:<10} {}'.format(multi, param.requires_grad))
        k += multi
    print('total params:', k)


def cmp_model_weights(model1, model2):
    for p1, p2 in zip(model1.params(), model2.params()):  # metamodule
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


model = ResNet32(num_classes=10)
sgd_model = load_model(model, ckpt_path='../output/asm_cifar10_imb1_s0.4_r1.0_Mar13_230920/asm_model.pth')
adam_model = load_model(model, '../output/hope_adamW_ams_cifar10_imb1_s0.4_r1.0_m300_Apr02_101739/asm_model.pth')

print_model_params(sgd_model)



# cmp_model_weights(sgd_model, adam_model)
