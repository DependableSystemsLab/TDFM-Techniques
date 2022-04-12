from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle


import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import random
import glob
from skimage import io
import pandas as pd
import math


class GTSRB(data.Dataset):
    def __init__(self, train=True, meta=False, num_meta=1000,
                 corruption_prob=0, transform=None, target_transform=None):
        self.root_dir = "/home/GTSRB/"
        self.train_root_dir = self.root_dir + "Final_Training/Images/"
        self.test_root_dir = self.root_dir + "Final_Test/"
        num_classes = 43

        if train:
            train_images, train_labels = self.__read_train_data(self.train_root_dir)
            self.images = train_images
            self.targets = train_labels
        else:
            test_images, test_labels = self.__read_test_data(self.test_root_dir)
            self.images = test_images
            self.targets = test_labels

        if meta is True:
            num = len(self.targets)
            meta_list = random.sample(range(num), num_meta)
            self.images = np.take(self.images, meta_list, axis=0)
            self.targets = np.take(self.targets, meta_list)
        elif train:
            self.targets = self.__train_label_noise(train_labels, corruption_prob)

        # Calculate len
        self.data_len = len(self.targets)

    def __getitem__(self, index):
        im_as_im = self.images[index]
        im_as_ten = np.moveaxis(im_as_im, -1, 0)
        label = self.targets[index]
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len

    def __read_train_data(self, train_root_dir):
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(train_root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            img = io.imread(img_path)
            label = self.__get_class(img_path)
            imgs.append(img)
            labels.append(label)

        train_images = np.array(imgs, dtype='float32')
        train_labels = np.array(labels)
        return train_images, train_labels

    def __read_test_data(self, test_root_dir):
        test = pd.read_csv(test_root_dir + "Labels/GT-final_test.csv", sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join(test_root_dir + "Images/", file_name)
            img = io.imread(img_path)

            x_test.append(img)
            y_test.append(class_id)

        test_images = np.array(x_test, dtype='float32')
        test_labels = np.array(y_test)
        return test_images, test_labels

    def __get_class(self, img_path):
            return int(img_path.split('/')[-2])

    def __train_label_noise(self, train_labels, mixing_ratio):
        y_train = train_labels
        num = y_train.shape[0]
        err_sz = mixing_ratio
        err_sz = err_sz * num
        err_sz = math.floor(err_sz)
        ind = random.sample(range(num), err_sz)
        for item in ind:
            r = list(range(0, y_train[item])) + list(range(y_train[item] + 1, 10))
            y_train[item] = random.choice(r)
        return y_train
