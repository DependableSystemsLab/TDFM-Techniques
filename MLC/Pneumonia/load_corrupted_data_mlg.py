from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle


import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import datasets

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import random
import glob
from skimage import io
import pandas as pd
import math


class PneumoniaDataset(data.Dataset):
    def __init__(self, train=True, meta=False, num_meta=1000,
                 corruption_prob=0, transform=None, target_transform=None):
        self.pneumonia_root = "/home/Pneumonia/"
        self.train_root_dir = self.pneumonia_root + "train"
        self.test_root_dir = self.pneumonia_root + "test"
        num_classes = 2

        if train:
            trainset = datasets.ImageFolder(self.train_root_dir, transform = transform)
            self.images = [np.array(t[0]) for t in trainset]
            self.targets = [t[1] for t in trainset]
            self.images = np.array(self.images)
            self.targets = np.array(self.targets)

        else:
            testset = datasets.ImageFolder(self.test_root_dir, transform = transform)
            self.images = [np.array(t[0]) for t in testset]
            self.targets = [t[1] for t in testset]
            self.images = np.array(self.images)
            self.targets = np.array(self.targets)

        if meta is True:
            num = len(self.targets)
            meta_list = random.sample(range(num), num_meta)
            self.images = np.take(self.images, meta_list, axis=0)
            self.targets = np.take(self.targets, meta_list)
        elif train:
            self.targets = self.__train_label_noise(self.targets, corruption_prob)

        # Calculate len
        self.data_len = len(self.targets)

    def __getitem__(self, index):
        im_as_im = self.images[index]
        label = self.targets[index]
        return (im_as_im, label)

    def __len__(self):
        return self.data_len

    def __train_label_noise(self, train_labels, mixing_ratio):
        y_train = train_labels
        num = y_train.shape[0]
        err_sz = mixing_ratio
        err_sz = err_sz * num
        err_sz = math.floor(err_sz)
        ind = random.sample(range(num), err_sz)
        for item in ind:
            maxval = np.amax(y_train)
            r = list(range(0, y_train[item])) + list(range(y_train[item] + 1, maxval+1))
            y_train[item] = random.choice(r)
        return y_train

