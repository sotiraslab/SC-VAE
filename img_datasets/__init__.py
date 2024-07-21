# Copyright (c) 2022-present, Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageNet, Caltech101, Caltech256
from torchvision import transforms, datasets
from img_datasets.miniimagenet import MiniImagenet

from .lsun import LSUNClass
from .ffhq import FFHQ
from .transforms import create_transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))

def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'imagenet':
        root = root if root else 'dataset/imagenet'
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet(root, split='val', transform=transforms_val)
    elif config.dataset.type == 'imagenet_u':
        root = root if root else 'dataset/imagenet'

        def target_transform(_):
            return 0
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn, target_transform=target_transform)
        dataset_val = ImageNet(root, split='val', transform=transforms_val, target_transform=target_transform)
    elif config.dataset.type == 'ffhq':
        root = root if root else 'dataset/FFHQ/resized'
        dataset_trn = FFHQ(root, split='train', transform=transforms_trn)
        dataset_val = FFHQ(root, split='val', transform=transforms_val)
    elif config.dataset.type in ['LSUN-cat', 'LSUN-church', 'LSUN-bedroom', 'LSUN-classroom']:
        root = root if root else 'dataset/lsun'
        category_name = config.dataset.type.split('-')[-1]
        dataset_trn = LSUNClass(root, category_name=category_name, split='train', transform=transforms_trn)
        dataset_val = LSUNClass(root, category_name=category_name, split='val', transform=transforms_val)
    elif config.dataset.type in ['caltech101', 'caltech256']:
        if config.dataset.type == 'caltech101':
            root = root if root else 'dataset/caltech101'
            #category_name = config.dataset.type.split('-')[-1]
            dataset_trn = Caltech101(root, transform=transforms_trn, download = True)
            dataset_val = Caltech101(root, transform=transforms_val)
        elif config.dataset.type == 'caltech256':
            root = root if root else 'dataset/'
            #category_name = config.dataset.type.split('-')[-1]
            dataset_trn = Caltech256(root, transform=transforms_trn, download = True)
            dataset_val = Caltech256(root, transform=transforms_val)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)

    if config.experiment.smoke_test:
        dataset_len = int(len(dataset_trn)/10)
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = torch.utils.data.Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val


def create_dataset_loader(args, model_config):
    # load dataset
    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                                           download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False, download=True,
                                          transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                                                  train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                                                 train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                                             train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                                            train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True, download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True, download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True, download=True, transform=transform)
        num_channels = 3
    elif args.dataset == 'FFHQ':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'imagenet':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'LSUN-cat':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'LSUN-church':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'LSUN-bedroom':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'cc3m':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'caltech101':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    elif args.dataset == 'caltech256':
        train_dataset, valid_dataset = create_dataset(model_config, is_eval=args.eval)
        test_dataset = valid_dataset
        num_channels = 3
    else:
        raise NotImplementedError('%s not implemented..' % args.dataset)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=model_config.experiment.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=model_config.experiment.batch_size, shuffle=False,
                                               drop_last=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32, shuffle=False)

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader, num_channels