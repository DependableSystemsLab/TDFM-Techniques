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
    from load_corrupted_data_mlg import PneumoniaDataset
    PNEUMONIA_MEAN = [0.485, 0.456, 0.406]
    PNEUMONIA_STD = [0.229, 0.224, 0.225]

    if True: # no augment as used by mwnet
        train_transform = transforms.Compose([
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(PNEUMONIA_MEAN, PNEUMONIA_STD)])
    else:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(PNEUMONIA_MEAN, PNEUMONIA_STD)
        ])
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(PNEUMONIA_MEAN, PNEUMONIA_STD)
    ])

    args.num_meta = int(5863 * gold_fraction)
    num_classes = 2

    if args.dataset == 'pneumonia':
        train_data_meta = PneumoniaDataset(
            train=True, meta=True, num_meta=args.num_meta, corruption_prob=corruption_prob,
            transform=train_transform)
        train_data = PneumoniaDataset(
            train=True, meta=False, num_meta=args.num_meta, corruption_prob=corruption_prob,
            transform=train_transform)
        test_data = PneumoniaDataset(train=False, transform=test_transform)

        valid_data = PneumoniaDataset(
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

