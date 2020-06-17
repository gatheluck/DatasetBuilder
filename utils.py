import os
import sys
import tqdm
import torch
import torchvision


def calculate_stats(path: str):
    """
    calculate statistics (mean and std) of dataset.

    Args
    - path: path to dataset.
    """
    dataset = torchvision.datasets.ImageFolder(
        path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=100)

    mean = 0.0
    std = 0.0

    # calculate mean
    with tqdm.tqdm(loader) as pbar:
        for img, _ in pbar:
            img = img.view(img.size(0), img.size(1), -1)  # merge h and w channels and become (b, c, h*w)
            mean += img.mean(dim=2).sum(dim=0)  # take mean and become (c)
        mean /= len(dataset)

    # calculate std
    var = 0.0
    with tqdm.tqdm(loader) as pbar:
        for img, _ in pbar:
            img = img.view(img.size(0), img.size(1), -1)  # merge h and w channels and become (b, c, h*w)
            var += ((img - mean[None, :, None]) ** 2).mean(dim=2).sum(dim=0)  # calculate variance (c)
        std = torch.sqrt(var / len(dataset))

    return mean, std


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, required=True, help='path to dataset')
    opt = parser.parse_args()

    print('calculate_stats of path: {}'.format(opt.datasetpath))
    mean, std = calculate_stats(opt.datasetpath)
    print('mean: {}, std: {}'.format(mean, std))
