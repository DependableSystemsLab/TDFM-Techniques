import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from dataiterator import DataIterator

def prepare_data(gold_fraction, corruption_prob, corruption_type, args):
    return prepare_data_mwnet(gold_fraction, corruption_prob, corruption_type, args)


def prepare_data_mwnet(gold_fraction, corruption_prob, corruption_type, args):
    from load_corrupted_data_mlg import GTSRB
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if True: # no augment as used by mwnet
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    args.num_meta = int(39209 * gold_fraction)
    num_classes = 43

    if args.dataset == 'gtsrb':
        train_data_meta = GTSRB(
            train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            transform=train_transform)
        train_data = GTSRB(
            train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            transform=train_transform)
        test_data = GTSRB(train=False, transform=test_transform)

        valid_data = GTSRB(
            train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            transform=train_transform)

    train_gold_loader = DataIterator(torch.utils.data.DataLoader(train_data_meta, batch_size=args.bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True))
    train_silver_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.bs, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_gold_loader, train_silver_loader, valid_loader, test_loader, num_classes

