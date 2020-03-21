from torch.utils.data import Dataset


class CIFAR(Dataset):
    def __init__(self, data, targets,
                 transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)  # images

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # np img (H x W x C)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
