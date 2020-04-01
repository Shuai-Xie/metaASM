import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import CIFAR
from datasets.transforms import transform_test, transform_train
from utils.base_utils import to_numpy, to_var
import numpy as np


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
    model.eval()  # 自动固定BN，不会取平均，而是使用训练好的值
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
        y_pred_prob = np.vstack((y_pred_prob, probs))

    return y_pred_prob


"""
for batch unlabel imgs
"""


def cal_pred_acc(preds, idxs, targets):
    if len(idxs) > 0:
        gts = np.take(targets, idxs, axis=0)
        acc = sum(preds == gts) / len(gts)
        return acc
    return 0


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
def asm_split_hc_delta_uc_K(model, input, target,
                            hc_delta,
                            uc_select_fn, K):
    """
    input: torch.Size([100, 32, 32, 3]), torch.uint8, HWC
    target: torch.Size([100]), torch.int64
    """
    model.eval()
    # np
    input, target = input.numpy().astype('uint8'), target.numpy().astype('int64')

    # tensor
    input_var = cvt_input_var(input, train=False)  # transfrom test

    # infer on batch input unlabel data
    outputs = model(input_var)
    probs = F.softmax(outputs, dim=-1)
    probs = to_numpy(probs)  # =0?

    # hc
    hc_idxs, hc_preds = get_hc_samples(probs, hc_delta)

    # uc
    ucs, tmp_uc_idxs = uc_select_fn(probs, K)
    tmp_uc_preds = ucs[:, 2].astype(int)
    uc_idxs, uc_preds = [], []

    for idx, pred in zip(tmp_uc_idxs, tmp_uc_preds):
        if idx not in hc_idxs:
            uc_idxs.append(idx)
            uc_preds.append(pred)

    # acc
    hc_acc = cal_pred_acc(hc_preds, hc_idxs, target)
    uc_acc = cal_pred_acc(uc_preds, uc_idxs, target)

    # ratio
    hc_ratio = len(hc_idxs) / len(probs)
    uc_ratio = len(uc_idxs) / len(probs)

    # data
    uc_data, uc_targets = input[uc_idxs], target[uc_idxs]
    hc_data, hc_targets = input[hc_idxs], hc_preds

    return {
        'hc': {
            'data': hc_data,
            'targets': hc_targets,
            'ratio': hc_ratio,
            'idxs': hc_idxs,
            'acc': hc_acc,
        },
        'uc': {
            'data': uc_data,
            'targets': uc_targets,
            'ratio': uc_ratio,
            'idxs': uc_idxs,  # save to remap to global uc_idxs
            'acc': uc_acc,
        }
    }


@torch.no_grad()
def asm_split_hc_uc_delta(model, input, target,
                          hc_delta=0.5,
                          uc_delta=1.27):
    """
    input: torch.Size([100, 32, 32, 3]), torch.uint8, HWC
    target: torch.Size([100]), torch.int64
    """
    model.eval()
    # tensor -> np
    input, target = input.numpy().astype('uint8'), target.numpy().astype('int64')

    # np -> model input tensor
    input_var = cvt_input_var(input, train=False)  # transfrom test

    # infer on batch input unlabel data
    outputs = model(input_var)
    probs = F.softmax(outputs, dim=-1)
    probs = to_numpy(probs)  # =0?

    results = get_hc_uc_samples(probs, hc_delta, uc_delta)

    hc_idxs, hc_preds = results['hc']['idxs'], results['hc']['preds']
    uc_idxs, uc_preds = results['uc']['idxs'], results['uc']['preds']

    hc_ratio = len(hc_idxs) / len(probs)
    uc_ratio = len(uc_idxs) / len(probs)

    hc_acc = cal_pred_acc(hc_preds, hc_idxs, target)
    uc_acc = cal_pred_acc(uc_preds, uc_idxs, target)

    uc_data, uc_targets = input[uc_idxs], target[uc_idxs]
    hc_data, hc_targets = input[hc_idxs], hc_preds

    return {
        'hc': {
            'data': hc_data,
            'targets': hc_targets,
            'ratio': hc_ratio,
            'idxs': hc_idxs,  # not use
            'acc': hc_acc,
        },
        'uc': {
            'data': uc_data,
            'targets': uc_targets,
            'ratio': uc_ratio,
            'idxs': uc_idxs,  # save to remap to global uc_idxs
            'acc': uc_acc
        }
    }


"""
4 ways to select informative samples by prob vector
"""


# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return None, np.random.choice(range(len(y_pred_prob)), n_samples)


# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)  # lc is max_prob
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index, max_prob, pred_label))
    lci = lci[lci[:, 1].argsort()]  # 越靠前 max_prob 越小
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
    origin_index = np.arange(0, len(y_pred_prob))
    entro = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)  # treat NaNs as 0!
    pred_label = np.argmax(y_pred_prob, axis=1)

    # 保留 ori_idxs, en, pred; argsort 已将 0 带入排序了，ori_idx 不受影响
    eni = np.column_stack((origin_index, entro, pred_label))
    eni = eni[(-eni[:, 1]).argsort()]  # 越靠前 -en 越小，en 越大，混乱度越高，越不确定
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
    hcs = eni[(eni[:, 1] > 0) & (eni[:, 1] < delta)]  # valid & hc
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)  # idx, pred_labels


def get_hc_uc_samples(y_pred_prob, hc_delta=0.5, uc_delta=1.27):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    eni = eni[eni[:, 1] > 0]  # valid
    hcs = eni[eni[:, 1] < hc_delta]  # hc
    ucs = eni[eni[:, 1] > uc_delta]  # uc
    return {
        'hc': {
            'idxs': hcs[:, 0].astype(int),
            'preds': hcs[:, 2].astype(int),
        },
        'uc': {
            'idxs': ucs[:, 0].astype(int),
            'preds': ucs[:, 2].astype(int),
        }
    }


def cal_entropy(max_prob, num_classes):
    assert 0 <= max_prob <= 1
    other_cls_prob = (1 - max_prob) / (num_classes - 1)
    prob = np.array([max_prob] + [other_cls_prob] * (num_classes - 1))
    print(np.sum(prob))
    entro = -np.nansum(np.multiply(prob, np.log(prob)), axis=0)  # nan -> 0
    print(entro)


def see_entroy():
    # prob = np.array([0.9991] + [0.0001] * 9) # 0.00918890121322387
    # prob = np.array([0.991] + [0.001] * 9)  # 0.07112917546111895
    # prob = np.array([0.91] + [0.01] * 9)  # 0.5002880350577578
    # prob = np.array([0.64] + [0.04] * 9)  # 1.4444190426347405
    prob = np.array([0.73] + [0.03] * 9)  # 1.4444190426347405
    # prob = np.array([0.91] + [0.01] * 99)  # 4.644941202447039
    print(np.sum(prob))
    # nansum, RuntimeWarning, not break
    entro = -np.nansum(np.multiply(prob, np.log(prob)), axis=0)  # nan -> 0
    print(entro)


if __name__ == '__main__':
    # see_entroy()
    cal_entropy(0.7, num_classes=10)
