import os
import sys
import hydra
import omegaconf
import torch
import torchvision


class DatasetBuilder(object):
    SUPPORTED_DATASET = set('svhn cifar10 imagenet100 imagenet'.split())

    def __init__(self, **kwargs):
        """
        Args
        - name (str)       : name of dataset
        - input_size (int) : input image size
        - mean (tuple)     : mean of normalized pixel value of channels
        - std (tuple)      : standard deviation of normalized pixel value of channels
        - root_path (str)  : root path to dataset
        """
        required_keys = set('name input_size mean std root_path'.split())
        parsed_args = self._parse_args(required_keys, kwargs)

        for k, v in parsed_args.items():
            print('{}:{}'.format(k, v))
            setattr(self, k, v)

        self.root_path = os.path.join(self.root_path, self.name)

        assert self.name in self.SUPPORTED_DATASET, 'name of dataset is invalid.'
        assert self.input_size > 0, 'input_size should be larger than 0.'
        assert len(self.mean) == len(self.std), 'length of mean and std should be same.'

    def _parse_args(self, required_keys: set, input_args: dict) -> dict:
        """
        parse input args
        Args
        - required_keys (set) : set of required keys for input_args
        - input_args (dict)   : dict of input arugments
        """
        parsed_args = dict()

        for k in required_keys:
            if k not in input_args.keys():
                raise ValueError('initial args are invalid.')
            else:
                parsed_args[k] = input_args[k]

        return parsed_args

    def __call__(self, train: bool, normalize: bool, binary_target: int = None, optional_transform=[]):
        """
        Args
        - train (bool)              : use train set or not.
        - normalize (bool)          : do normalize or not.
        - binary_target (int)       : if not None, creates datset for binary classification.
        - optional_transform (list) : list of optional transformations. these are applied before normalization.
        """
        transform = self._get_transform(self.name, self.input_size, self.mean, self.std, train, normalize, optional_transform)

        # get dataset
        if self.name == 'svhn':
            dataset = torchvision.datasets.SVHN(root=self.root_path, split='train' if train else 'test', transform=transform, download=True)
            targets_name = 'labels'
        elif self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
            targets_name = 'targets'
        elif self.name in 'imagenet100 imagenet'.split():
            root = os.path.join(self.root_path, 'train' if train else 'val')
            dataset = torchvision.datasets.ImageFolder(root, transform=transform)
        else:
            raise NotImplementedError

        # make binary classification dataset
        if binary_target is not None:
            dataset = self._binarize_dataset(dataset, targets_name, binary_target)

        return dataset

    def _binarize_dataset(self, dataset, targets_name: str, binary_target: int):
        """
        Args
        - dataset             : pytorch dataset class.
        - targets_name (str)  : intermediate variable to compensate inconsistent torchvision API.
        - binary_target (int) : true class label of binary classification.
        """
        targets = getattr(dataset, targets_name)
        assert 0 <= binary_target <= max(targets)

        targets = [1 if target == binary_target else 0 for target in targets]
        setattr(dataset, targets_name, targets)

        return dataset

    def _get_transform(self, name: str, input_size: int, mean: tuple, std: tuple, train: bool, normalize: bool, optional_transform=[]):
        """
        Args
        - name (str)                : name of dataset.
        - input_size (int)          : input image size.
        - mean (tuple)              : mean of normalized pixel value of channels
        - std (tuple)               : standard deviation of normalized pixel value of channels
        - train (bool)              : use train set or not.
        - normalize (bool)          : normalize image or not.
        - optional_transform (list) : list of optional transformations. these are applied before normalization.
        """
        transform = []

        # arugmentation
        # imagenet100 / imagenet
        if input_size == 224:
            if train:
                transform.extend([
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            else:
                transform.extend([
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                ])
        # cifar10 / svhn
        elif input_size == 32:
            if train:
                transform.extend([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                ])
            else:
                pass
        else:
            raise NotImplementedError

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(), ])

        # optional (Fourier Noise, Patch Shuffle, etc.)
        if optional_transform:
            transform.extend(optional_transform)

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=mean, std=std),
            ])

        return torchvision.transforms.Compose(transform)


@hydra.main(config_path='./conf/config.yaml')
def test(cfg: omegaconf.DictConfig):
    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), './data'), **cfg.dataset)
    test_set = dataset_builder(train=False, normalize=True)
    # print(test_set.targets)

    train_set = dataset_builder(train=True, normalize=True, binary_target=7)
    # print(test_set.targets)


if __name__ == '__main__':
    test()
