import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

import tqdm
import click
import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchvision
from typing import List

from libs.metric import accuracy
from libs.logger import Logger


def evaluate_corruption_accuracy(
    model,
    dataset_builder,
    log_dir: str,
    num_samples: int,
    corruptions: list,
    batch_size: int,
    device: str,
    **kwargs,
):
    """
    """

    if (device != "cpu") and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    log_path = os.path.join(log_dir, os.path.join("corruption_result.csv"))
    logger = Logger(path=log_path, mode="test")

    with tqdm.tqdm(total=len(corruptions), ncols=80) as pbar:
        for i, corruption_type in enumerate(corruptions):
            dataset = dataset_builder(
                train=False,
                normalize=True,
                num_samples=num_samples,
                corruption_type=corruption_type,
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=False
            )

            accuracies = list()

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autograd.no_grad():
                    y_predict_std = model(x)

                stdacc1, stdacc5 = accuracy(y_predict_std, y, topk=(1, 5))
                accuracies.append(stdacc1.item())

            log_dict = collections.OrderedDict()
            log_dict["corruption_type"] = corruption_type
            log_dict["accuracy"] = sum(accuracies) / float(len(accuracies))
            logger.log(log_dict)

            pbar.set_postfix(
                collections.OrderedDict(
                    corruption_type="{}".format(corruption_type),
                    acc="{}".format(log_dict["accuracy"]),
                )
            )
            pbar.update()

    df = pd.read_csv(log_path)
    result_dict = dict(zip(df["corruption_type"], df["accuracy"]))
    mean_corruption_acc = sum(result_dict.values()) / float(len(result_dict))
    create_barplot(
        result_dict,
        title="mean corruption acc: {0:0.1f}".format(mean_corruption_acc),
        savepath=os.path.join(log_dir, "plot_result.png"),
    )


def evaluate_imagenet_c(
    model,
    transform,
    dataset_dir: str,
    log_dir: str,
    corruptions: List[str],
    batch_size: int,
    device: str,
    **kwargs,
) -> None:
    """
    Evaluate corruption accuracy on ImageNet-C.
    """

    if (device != "cpu") and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    log_path = os.path.join(log_dir, os.path.join("imagenet_c_result.csv"))
    logger = Logger(path=log_path, mode="test")

    with tqdm.tqdm(total=len(corruptions), ncols=80) as pbar:
        for i, corruption_type in enumerate(corruptions):
            accuracies = list()

            for j in range(0, 6):  # imagenet-c dataset is separated to 5 small sets.
                datasetpath = os.path.join(dataset_dir, corruption_type, str(j))
                dataset = torchvision.datasets.ImageFolder(datasetpath, transform)
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                )

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    with torch.autograd.no_grad():
                        y_predict_std = model(x)

                    stdacc1, stdacc5 = accuracy(y_predict_std, y, topk=(1, 5))
                    accuracies.append(stdacc1.item())

            log_dict = collections.OrderedDict()
            log_dict["corruption_type"] = corruption_type
            log_dict["accuracy"] = sum(accuracies) / float(len(accuracies))
            logger.log(log_dict)

            pbar.set_postfix(
                collections.OrderedDict(
                    corruption_type="{}".format(corruption_type),
                    acc="{}".format(log_dict["accuracy"]),
                )
            )
            pbar.update()

    df = pd.read_csv(log_path)
    result_dict = dict(zip(df["corruption_type"], df["accuracy"]))
    mean_corruption_acc = sum(result_dict.values()) / float(len(result_dict))
    create_barplot(
        result_dict,
        title="mean corruption acc: {0:0.1f}".format(mean_corruption_acc),
        savepath=os.path.join(log_dir, "plot_result.png"),
    )


def evaluate_cifar_c(
    model,
    dataset: str,
    dataset_dir: str,
    log_dir: str,
    corruptions: List[str],
    batch_size: int,
    device: str,
    **kwargs,
) -> None:
    """
    Evaluate corruption accuracy on CIFAR10-C or CIFAR100-C.
    """

    if (device != "cpu") and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    log_path = os.path.join(log_dir, os.path.join("cifar_c_result.csv"))
    logger = Logger(path=log_path, mode="test")

    with tqdm.tqdm(total=len(corruptions), ncols=80) as pbar:
        for i, corruption_type in enumerate(corruptions):
            accuracies = list()

            dataset.data = np.load(dataset_dir + corruption_type + ".npy")
            dataset.targets = torch.LongTensor(np.load(dataset_dir + "labels.npy"))

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autograd.no_grad():
                    y_predict_std = model(x)

                stdacc1, stdacc5 = accuracy(y_predict_std, y, topk=(1, 5))
                accuracies.append(stdacc1.item())

            log_dict = collections.OrderedDict()
            log_dict["corruption_type"] = corruption_type
            log_dict["accuracy"] = sum(accuracies) / float(len(accuracies))
            logger.log(log_dict)

            pbar.set_postfix(
                collections.OrderedDict(
                    corruption_type="{}".format(corruption_type),
                    acc="{}".format(log_dict["accuracy"]),
                )
            )
            pbar.update()

    df = pd.read_csv(log_path)
    result_dict = dict(zip(df["corruption_type"], df["accuracy"]))
    mean_corruption_acc = sum(result_dict.values()) / float(len(result_dict))
    create_barplot(
        result_dict,
        title="mean corruption acc: {0:0.1f}".format(mean_corruption_acc),
        savepath=os.path.join(log_dir, "plot_result.png"),
    )


def create_barplot(accs: dict, title: str, savepath: str):
    y = list(accs.values())
    x = np.arange(len(y))
    xticks = list(accs.keys())

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f"{j:.1f}", ha="center", va="bottom", fontsize=7)

    plt.title(title)
    plt.ylabel("Accuracy (%)")

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis="y")
    plt.savefig(savepath)
    plt.close()


if __name__ == "__main__":

    def _get_model(
        arch: str, num_classes: int, device: str = "cuda"
    ) -> torch.nn.Module:
        if arch == "resnet50":
            model = torchvision.models.resnet50(num_classes=num_classes)
        else:
            raise NotImplementedError

        return model.to(device)

    def _get_transform(input_size: int, dataset_name: str):
        _means = {
            "cifar10-c": [0.49139968, 0.48215841, 0.44653091],
            "cifar100-c": [0.50707516, 0.48654887, 0.44091784],
            "imagenet-c": [0.485, 0.456, 0.406],
        }
        _stds = {
            "cifar10-c": [0.24703223, 0.24348513, 0.26158784],
            "cifar100-c": [0.26733429, 0.25643846, 0.27615047],
            "imagenet-c": [0.229, 0.224, 0.225],
        }

        mean, std = _means[dataset_name], _stds[dataset_name]

        transform = [
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]

        return torchvision.transforms.Compose(transform)

    _corruptions = "gaussian_noise shot_noise speckle_noise impulse_noise defocus_blur gaussian_blur motion_blur zoom_blur snow fog brightness contrast elastic_transform pixelate jpeg_compression spatter saturate frost".split()

    # parse argument
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-a", "--arch", type=str, required=True)
    parser.add_argument("-w", "--weight", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-s", "--input_size", type=int, required=True)
    parser.add_argument("-n", "--num_classes", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("-l", "--log_dir", type=str, required=True)
    opt = parser.parse_args()

    # create logdir
    os.makedirs(opt.log_dir, exist_ok=True)

    # get model
    model = _get_model(opt.arch, opt.num_classes)
    model.load_state_dict(torch.load(opt.weight))

    # get transform for dataset
    transform = _get_transform(opt.input_size, opt.dataset)

    if opt.dataset in ["cifar10-c", "cifar100-c"]:
        if opt.dataset == "cifar10-c":
            dataset = torchvision.datasets.CIFAR10(
                "../data/cifar", train=False, transform=transform, download=True
            )
        elif opt.dataset == "cifar100-c":
            dataset = torchvision.datasets.CIFAR100(
                "../data/cifar", train=False, transform=transform, download=True
            )
        else:
            raise NotImplementedError
        evaluate_cifar_c(
            model,
            dataset,
            opt.dataset_dir,
            opt.log_dir,
            _corruptions,
            opt.batch_size,
            "cuda",
        )
    elif opt.dataset in ["imagenet-c"]:
        evaluate_imagenet_c(
            model,
            transform,
            opt.dataset_dir,
            opt.log_dir,
            _corruptions,
            opt.batch_size,
            "cuda",
        )
    else:
        raise NotImplementedError
