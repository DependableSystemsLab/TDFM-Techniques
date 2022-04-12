from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets

from lr.models.models_meta import StringEnum


from src import tfi, config

def perform_data_fi(
    train_images,
    train_labels,
    final_fault):

    conf_file = "/home/confFiles/" + final_fault + ".yaml"

    if (final_fault.startswith("label_err")):
        tf_res = tfi.inject(y_test=train_labels, confFile=conf_file)
        train_labels = tf_res
    else:
        train_images, train_labels = tfi.inject(x_test=train_images,y_test=train_labels, confFile=conf_file)

    print("\n\nLength of labels:  " + str(len(train_labels)) + "\n\n")

    return train_images, train_labels


class Dataset(StringEnum):
    CIFAR_10 = "cifar10"
    CIFAR_100 = "cifar100"
    IMAGENET = "imagenet"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    GTSRB = "gtsrb"
    PNEUMONIA = "pneumonia"


def get_dataset_type_by_name(dataset_name):
    if dataset_name == Dataset.CIFAR_10.value:
        return Dataset.CIFAR_10
    elif dataset_name == Dataset.CIFAR_100.value:
        return Dataset.CIFAR_100
    elif dataset_name == Dataset.IMAGENET.value:
        return Dataset.IMAGENET
    elif dataset_name == Dataset.MNIST.value:
        return Dataset.MNIST
    elif dataset_name == Dataset.FASHION_MNIST.value:
        return Dataset.FASHION_MNIST
    elif dataset_name == Dataset.GTSRB.value:
        return Dataset.GTSRB
    elif dataset_name == Dataset.PNEUMONIA.value:
        return Dataset.PNEUMONIA
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))


def get_dataset_by_type(dataset_type, seed, final_fault=""):
    if dataset_type == Dataset.CIFAR_10:
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        if final_fault != "golden":
            x_train, y_train = perform_data_fi(x_train, y_train, final_fault)
        num_classes = 10
        test_size = 1 / 6
    elif dataset_type == Dataset.CIFAR_100:
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        num_classes = 100
        test_size = 1 / 6
    elif dataset_type == Dataset.IMAGENET:
        raise NotImplementedError("ImageNet not provided yet.")
    elif dataset_type == Dataset.MNIST:
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # Add channel dimension
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        num_classes = 10

        test_size = 1 / 7
    elif dataset_type == Dataset.FASHION_MNIST:
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
        num_classes = 10

        # Add channel dimension
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        test_size = 1 / 7
    elif dataset_type == Dataset.GTSRB:
        (x_train, y_train), (x_test, y_test) = load_gtsrb()
        if final_fault != "golden":
            x_train, y_train = perform_data_fi(x_train, y_train, final_fault)
        num_classes = 43
    elif dataset_type == Dataset.PNEUMONIA:
        (x_train, y_train), (x_test, y_test) = load_pneumonia()
        if final_fault != "golden":
            x_train, y_train = perform_data_fi(x_train, y_train, final_fault)
        num_classes = 2

    else:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))

    return (x_train, y_train), (x_test, y_test), num_classes


def load_gtsrb():
    root_dir = "/home/GTSRB/"
    train_root_dir = root_dir + "Final_Training/Images/"
    test_root_dir = root_dir + "Final_Test/"

    import os
    import glob
    from skimage import io
    import pandas as pd

    def __read_train_data(train_root_dir):
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(train_root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            img = io.imread(img_path)
            label = __get_class(img_path)
            imgs.append(img)
            labels.append(label)

        train_images = np.array(imgs, dtype='float32')
        train_labels = np.array(labels)
        return train_images, train_labels

    def __read_test_data(test_root_dir):
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

    def __get_class(img_path):
            return int(img_path.split('/')[-2])

    train_images, train_labels = __read_train_data(train_root_dir)
    test_images, test_labels = __read_test_data(test_root_dir)

    return (train_images, train_labels), (test_images, test_labels)


def load_pneumonia():
    pneumonia_root = "/home/Pneumonia/"
    train_root_dir = pneumonia_root + "train"
    test_root_dir = pneumonia_root + "test"

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import glob
    from skimage import io

    def __read_train_data(root_dir):
        pixel_size = 128
        target_size = (pixel_size, pixel_size)
        batch_size = 64
        train_datagen = ImageDataGenerator(
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        training_set = train_datagen.flow_from_directory(root_dir,
                                                 target_size = target_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'sparse',
                                                 color_mode='grayscale',
                                                 shuffle=False)
        return __get_data_label(training_set)

    def __read_test_data(root_dir):
        imgs = []
        labs = []

        all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*/*.jpeg')))
        print(all_img_paths)
        for img_path in all_img_paths:
            img = io.imread(img_path)
            label = __get_class(img_path)
            imgs.append(img)
            labs.append(label)

        images = np.array(imgs, dtype='float32')
        labels = np.array(labs, dtype='int')
        images = np.expand_dims(images, axis=3)
        return images, labels

    def __get_class(img_path):
        img_class = img_path.split('/')[-2]
        return 0 if img_class == "NORMAL" else 1

    def __get_data_label(training_generator):
        batch_index = 0
        while batch_index <= training_generator.batch_index:
            data = training_generator.next()
            if batch_index == 0:
                data_list = data[0]
                label_list = data[1]
            else:
                data_list = np.concatenate((data_list, data[0]), axis=0)
                label_list = np.concatenate((label_list, data[1]))
            batch_index = batch_index + 1
            label_list = label_list.astype(int)
        return data_list, label_list

    train_images, train_labels = __read_train_data(train_root_dir)
    test_images, test_labels = __read_test_data(test_root_dir)

    return (train_images, train_labels), (test_images, test_labels)
