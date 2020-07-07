import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import PIL
import numpy as np
import torchvision


class CIFAR10C(torchvision.datasets.VisionDataset):
    SUPPORTED_CORRUPTIONS = set('gaussian_noise shot_noise speckle_noise impulse_noise defocus_blur gaussian_blur motion_blur zoom_blur snow fog brightness contrast elastic_transform pixelate jpeg_compression spatter saturate frost'.split())

    def __init__(self, root: str, corruption_type: str, transform=None, target_transform=None):
        assert corruption_type in self.SUPPORTED_CORRUPTIONS, 'unsupported corruption_type value'
        super(CIFAR10C, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data = np.load(os.path.join(root, corruption_type + '.npy'))
        self.targets = np.load(os.path.join(root, 'labels.npy'))

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torch

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    x_list = list()

    for corruption_type in set('gaussian_noise shot_noise speckle_noise impulse_noise defocus_blur gaussian_blur motion_blur zoom_blur snow fog brightness contrast elastic_transform pixelate jpeg_compression spatter saturate frost'.split()):
        dataset = CIFAR10C(root='../data/cifar10c', corruption_type=corruption_type, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        for x, t in loader:
            x_split = torch.split(x, 1)
            x_cat = torch.cat(x_split, dim=2)
            x_list.append(x_cat)
            break

    x_save = torch.cat(x_list, dim=-1)
    torchvision.utils.save_image(x_save, '../logs/cifar10c_samples.png')
