import torchvision.transforms as transforms
import torch.nn.functional as F

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
    transforms.ToTensor(),  # np 255(uint8) -> tensor 1.(float32), HWC -> CHW
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),  # 4D, same to np pad
                                      (4, 4, 4, 4), mode='reflect').squeeze()),  # recover 3D
    transforms.ToPILImage(),  # C x H x W tensor
    transforms.RandomCrop(32),  # 32x32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # will div(255) to [0,1], 可以从 normal 的 mean/std 值设置知道
    normalize
])
