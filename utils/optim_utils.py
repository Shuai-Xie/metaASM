from optim.padam import Padam
from net.resnet import ResNet32
from pprint import pprint

model = ResNet32(10)

optimizer = Padam(model.params(),
                  lr=1e-1,
                  betas=(0.9, 0.999), eps=1e-8, partial=1 / 8)

param_groups = optimizer.param_groups  # list, len=1
for e in param_groups:
    print(e.keys())

