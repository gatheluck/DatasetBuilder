import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import tqdm
import click
import collections
import torch

from libs.metric import accuracy
from libs.logger import Logger


def evaluate_corruption_accuracy(model, dataset_builder, log_dir: str, corruptions: list, batch_size: int, device: str, **kwargs):
    """
    """

    if (device != 'cpu') and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model)

    log_path = os.path.join(log_dir, os.path.join('corruption_result.csv'))
    logger = Logger(path=log_path, mode='test')

    with tqdm.tqdm(total=len(corruptions), ncols=80) as pbar:
        for i, corruption_type in enumerate(corruptions):
            dataset = dataset_builder(train=False, normalize=True, corruption_type=corruption_type)
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
