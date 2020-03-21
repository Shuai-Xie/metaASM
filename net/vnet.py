import torch
import torch.nn as nn
from net.meta_modules import MetaModule, MetaLinear
import numpy as np
from utils import to_numpy


class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)  # 可计算 0 项，多少 neuron dead
        out = self.linear2(x)
        return out.sigmoid_()


@torch.no_grad()
def get_vnet_curve(vnet, upper_loss=10, num_pts=100, xy_path=None):
    vnet.eval()
    # x: loss, y: weight
    x = torch.linspace(0, upper_loss, steps=num_pts).reshape((num_pts, 1)).cuda()
    y = vnet(x)
    x, y = to_numpy(x).squeeze(), to_numpy(y).squeeze()  # (100,)
    # save npy
    if xy_path is not None:
        np.save(xy_path, (x, y))
    return x, y
