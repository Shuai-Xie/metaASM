import torch
import torch.nn.functional as F
from datasets.cls_datasets import CIFAR
from torch.utils.data import DataLoader
from datasets.imb_data_utils import transform_test
from utils import to_numpy, to_var
import numpy as np


class Delta_Scheduler:
    def __init__(self, init_val, min_val, max_steps):
        self.delta = init_val
        self.step_decay = (init_val - min_val) / max_steps

    def step(self):
        self.delta -= self.step_decay


@torch.no_grad()
def detect_unlabel_imgs(model, batch_unlabel_imgs, num_classes, bs=1):
    model.eval()
    # pass fake targets to build dataloader
    dataset = CIFAR(batch_unlabel_imgs, [-1] * len(batch_unlabel_imgs), transform=transform_test)
    dataloader = DataLoader(dataset, batch_size=bs,
                            shuffle=False, num_workers=4, drop_last=False)

    y_pred_prob = np.empty((0, num_classes))

    for batch_idx, (input, target) in enumerate(dataloader):
        input_var = to_var(input, requires_grad=False)
        outputs = model(input_var)  # logtis
        probs = F.softmax(outputs, dim=-1)  # logits -> probs [100,10]
        probs = to_numpy(probs)
        y_pred_prob = np.concatenate((y_pred_prob, probs), axis=0)

    return y_pred_prob


# 4 ways to select informative samples by prob vector

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
    max_prob = np.max(y_pred_prob, axis=1)
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
def get_high_conf_samples(y_pred_prob, delta):
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
