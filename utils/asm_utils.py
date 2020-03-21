import torch
import torch.nn.functional as F
from datasets.cls_datasets import CIFAR
from torch.utils.data import DataLoader
from datasets.imb_data_utils import transform_test, transform_train
from utils.base_utils import to_numpy, to_var
import numpy as np
from PIL import Image


class Delta_Scheduler:
    def __init__(self, init_val, min_val, max_steps):
        self.delta = init_val
        self.step_decay = (init_val - min_val) / max_steps

    def step(self):
        self.delta -= self.step_decay


"""
for many unlabel imgs
"""


@torch.no_grad()
def detect_unlabel_imgs(model, unlabel_imgs, num_classes, bs=100):
    model.eval()
    # build dataloader for batch infer
    dataset = CIFAR(data=unlabel_imgs,
                    targets=[-1] * len(unlabel_imgs),  # fake targets
                    transform=transform_test)
    dataloader = DataLoader(dataset,
                            batch_size=bs,
                            shuffle=False,  # 按顺序获取 probs
                            num_workers=4,
                            drop_last=False)

    y_pred_prob = np.empty((0, num_classes))

    for batch_idx, (input, target) in enumerate(dataloader):
        input_var = to_var(input, requires_grad=False)
        outputs = model(input_var)  # logtis
        probs = F.softmax(outputs, dim=-1)  # logits -> probs [100,10]
        probs = to_numpy(probs)
        y_pred_prob = np.concatenate((y_pred_prob, probs), axis=0)

    return y_pred_prob


"""
for batch unlabel imgs
"""


def cvt_input_var(input, train=True):
    B, H, W, C = input.shape
    input_var = torch.empty([B, C, H, W])

    if train:
        for i in range(B):
            input_var[i] = transform_train(input[i])
    else:
        for i in range(B):
            input_var[i] = transform_test(input[i])

    return to_var(input_var, requires_grad=False)  # cuda


@torch.no_grad()
def asm_split_samples(model, input, target,
                      hc_delta,
                      uc_select_fn, K):
    """
    input: torch.Size([100, 32, 32, 3]), torch.uint8, HWC
    target: torch.Size([100]), torch.int64
    """
    # np
    input = input.numpy().astype('uint8')  # np img
    target = target.numpy().astype('int64')

    # tensor
    input_var = cvt_input_var(input, train=False)  # transfrom test

    # infer on batch input unlabel data
    outputs = model(input_var)
    probs = F.softmax(outputs, dim=-1)
    probs = to_numpy(probs)  # =0?

    # hc
    hc_idxs, hc_preds = get_hc_samples(probs, hc_delta)
    hc_ratio = len(hc_idxs) / len(probs)

    if len(hc_idxs) > 0:
        hc_gts = np.take(target, hc_idxs, axis=0)  # divide 0
        hc_acc = sum(hc_preds == hc_gts) / len(hc_gts)
    else:
        hc_acc = 0

    # todo: 考虑是否要选入除 hc 后剩余数据；选 uc 为了加快训练
    _, uc_idxs = uc_select_fn(probs, K)
    uc_idxs = [k for k in uc_idxs if k not in hc_idxs]  # list, rm uc in hc
    uc_ratio = len(uc_idxs) / len(probs)
    # 只要 hc_ratio > 1 - K / len(probs) [ori uc_ratio], 二者之和 = 1

    asm_inputs = np.take(input, uc_idxs + list(hc_idxs), axis=0)
    asm_targets = np.append(target[uc_idxs], hc_preds, axis=0)  # 注意 hc use preds!

    # shuffle uc/hc together
    idxs = np.random.permutation(len(asm_inputs))
    asm_inputs, asm_targets = asm_inputs[idxs], asm_targets[idxs]  # np img, targets

    return asm_inputs, asm_targets, hc_acc, hc_ratio, uc_ratio


"""
4 ways to select informative samples by prob vector
"""


# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return None, np.random.choice(range(len(y_pred_prob)), n_samples)


# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    """
    @param y_pred_prob: model outputs, (N,10) [logits, prob] 均可
    @param n_samples: choose number
    @return:
        lci: [ori_idx, prob, label]  # (N,3)
        lci_idx: [ori_idx]  # (N,) ori_idx sort by confidence
    """
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)  # lc is max_prob
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index, max_prob, pred_label))
    lci = lci[lci[:, 1].argsort()]  # 按照 prob 排序
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)

    msi = np.column_stack((origin_index, margim_sampling, pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))  # (N,10)
    entro = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    eni = np.column_stack((origin_index, entro, pred_label))
    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


# Choose select criterion function
def get_select_fn(criterion):
    if criterion == 'rs':
        fn = random_sampling
    elif criterion == 'lc':
        fn = least_confidence
    elif criterion == 'ms':
        fn = margin_sampling
    elif criterion == 'en':
        fn = entropy
    else:
        raise ValueError('no such criterion')
    return fn


# Rank high confidence samples by entropy
def get_hc_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    eni = eni[eni[:, 1] > 0]  # valid
    hcs = eni[eni[:, 1] < delta]  # en < delte
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)  # idx, pred_labels


def see_entroy():
    # prob = np.array([0.9991] + [0.0001] * 9) # 0.00918890121322387
    # prob = np.array([0.991] + [0.001] * 9)  # 0.07112917546111895
    # prob = np.array([0.91] + [0.01] * 9)  # 0.5002880350577578
    prob = np.array([0.91] + [0.01] * 99)  # 0.5002880350577578
    # prob = np.array([0] * 10)
    print(np.sum(prob))
    # nansum, RuntimeWarning, not break
    entro = -np.nansum(np.multiply(prob, np.log(prob)), axis=0)  # nan -> 0
    print(entro)


if __name__ == '__main__':
    see_entroy()
