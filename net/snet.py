import torch.nn as nn
from net.meta_modules import MetaModule, MetaLinear


class SNet(MetaModule):  # Select Net
    def __init__(self, input, hidden1, output):
        super(SNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)  # 增加网络稀疏性
        out = self.linear2(x)
        return out.relu_()  # hard selection
