import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import tqdm
import click
import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from libs.metric import accuracy
from libs.logger import Logger


def evaluate_corruption_accuracy(model, dataset_builder, log_dir: str, num_samples: int, corruptions: list, batch_size: int, device: str, **kwargs):
    """
    """

    if (device != 'cpu') and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    log_path = os.path.join(log_dir, os.path.join('corruption_result.csv'))
    logger = Logger(path=log_path, mode='test')

    with tqdm.tqdm(total=len(corruptions), ncols=80) as pbar:
        for i, corruption_type in enumerate(corruptions):
            dataset = dataset_builder(train=False, normalize=True, num_samples=num_samples, corruption_type=corruption_type)
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            accuracies = list()

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autograd.no_grad():
                    y_predict_std = model(x)

                stdacc1, stdacc5 = accuracy(y_predict_std, y, topk=(1, 5))
                accuracies.append(stdacc1.item())

            log_dict = collections.OrderedDict()
            log_dict['corruption_type'] = corruption_type
            log_dict['accuracy'] = sum(accuracies) / float(len(accuracies))
            logger.log(log_dict)

            pbar.set_postfix(collections.OrderedDict(corruption_type='{}'.format(corruption_type), acc='{}'.format(log_dict['accuracy'])))
            pbar.update()

    df = pd.read_csv(log_path)
    result_dict = dict(zip(df['corruption_type'], df['accuracy']))
    mean_corruption_acc = sum(result_dict.values()) / float(len(result_dict))
    create_barplot(result_dict, title='mean corruption acc: {0:0.1f}'.format(mean_corruption_acc), savepath=os.path.join(log_dir, 'plot_result.png'))


def create_barplot(accs: dict, title: str, savepath: str):
    y = list(accs.values())
    x = np.arange(len(y))
    xticks = list(accs.keys())

    plt.bar(x, y)
    for i, j in zip(x, y):
        plt.text(i, j, f'{j:.1f}', ha='center', va='bottom', fontsize=7)

    plt.title(title)
    plt.ylabel('Accuracy (%)')

    plt.ylim(0, 100)

    plt.xticks(x, xticks, rotation=90)
    plt.yticks(np.linspace(0, 100, 11))

    plt.subplots_adjust(bottom=0.3)
    plt.grid(axis='y')
    plt.savefig(savepath)
    plt.close()
