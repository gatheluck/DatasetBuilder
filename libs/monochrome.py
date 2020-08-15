import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import torchvision


class Monochrome(torchvision.datasets.VisionDataset):
    def __init__(self,
                 input_size: int = 32,
                 mean: list = [0.5, 0.5, 0.5],
                 std: list = [0.2, 0.2, 0.2],
                 train: bool = True,
                 num_train: int = 50000,
                 num_val: int = 5000,
                 target_label: int = 0,
                 **kwargs):
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.train = train
        self.target_label = target_label
        self.num_data = num_train if self.train else num_val
        self.dist = torch.distributions.normal.Normal(torch.tensor(self.mean), torch.tensor(self.std))

    def __getitem__(self, index):
        x = self.dist.sample().view(3, 1, 1).repeat(1, self.input_size, self.input_size)
        return x, self.target_label

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    dataset = Monochrome(mean=[0.1, 0.2, 0.3])
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=8)

    for x, t in loader:
        # print(x)
        print(x.shape)
        print(x.dtype)

        # print(t)
        print(t.shape)
        print(t.dtype)

        break

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='../data/cifar10', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=8)

    for x, t in loader:
        # print(x)
        print(x.shape)
        print(x.dtype)

        # print(t)
        print(t.shape)
        print(t.dtype)

        break
