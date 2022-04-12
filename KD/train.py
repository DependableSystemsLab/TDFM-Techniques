"""
Trains the teacher network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import math
from resnet import get_resnet_by_n
from vgg import get_vgg16, get_vgg11
from convnet import get_convnet, get_deconvnet
from mobilenet import get_mobilenet
from distiller import Distiller

import json
import uuid
pred_list = []
identifier = str(uuid.uuid4())

import argparse
from src import tfi, config

parser = argparse.ArgumentParser(description='Train model with fault params')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'gtsrb', 'pneumonia'], default='cifar10')
parser.add_argument('--model_type', type=str, default="")
parser.add_argument('--final_fault', type=str, default="")

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=2)
args = parser.parse_args()


# training parameters
batch_size = args.batch_size # orig paper trained all networks with batch_size=128
epochs = args.epochs
data_augmentation = True
num_classes = 10 #Default for CIFAR-10

# subtracting pixel mean improves accuracy
subtract_pixel_mean = True


def get_model_by_name(model_type, input_shape):
    if model_type == "ResNet50":
        model = get_resnet_by_n(9, input_shape, classes=num_classes)
    elif model_type == "ResNet18":
        model = get_resnet_by_n(3, input_shape, classes=num_classes)
    elif model_type == "VGG11":
        model = get_vgg11(input_shape, classes=num_classes)
    elif model_type == "ConvNet":
        model = get_convnet(input_shape, classes=num_classes)
    elif model_type == "DeconvNet":
        model = get_deconvnet(input_shape, classes=num_classes)
    elif model_type == "MobileNet":
        model = get_mobilenet(input_shape, classes=num_classes)
    else:
        model = get_vgg16(input_shape, classes=num_classes)
    return model


def load_training_data(dataset, student_run=False):
    global num_classes
    final_fault = args.final_fault
    model_type = args.model_type

    if dataset == "cifar10":
        # load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == "gtsrb":
        (x_train, y_train), (x_test, y_test) = load_gtsrb()
        num_classes = 43
    elif dataset == "pneumonia":
        (x_train, y_train), (x_test, y_test) = load_pneumonia()
        num_classes = 2
    else:
        # load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), dataset + '/' + model_type)
    fault_npy = final_fault + ".npy"

    if student_run is False: # load training data from scratch for teacher runs
        if final_fault:
            x_train, y_train = perform_data_fi(x_train, y_train, final_fault)

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            if not final_fault.startswith("label_err"):
                imagefile = os.path.join(save_dir, "image-" + fault_npy)
                np.save(imagefile, x_train)

            labelfile = os.path.join(save_dir, "labels-" + fault_npy)
            np.save(labelfile, y_train)

        else:
            print("Golden run")

    else: # load training data directly for student runs
        if not final_fault.startswith("label_err"):
            imagefile = os.path.join(save_dir, "image-" + fault_npy)
            x_train = np.load(imagefile)

        labelfile = os.path.join(save_dir, "labels-" + fault_npy)
        y_train = np.load(labelfile)

    return setup_data(x_train, y_train, x_test, y_test)


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


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 45:
        lr *= 0.5e-3
    elif epoch > 35:
        lr *= 1e-3
    elif epoch > 25:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def setup_data(x_train, y_train, x_test, y_test):
    # normalize data.
    x_train = np.array(x_train).astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # if subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    return (x_train, y_train), (x_test, y_test)


def get_trained_model(model, model_type, x_train, y_train, x_test, y_test, data_augment=True):

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['acc'])
    model.summary()

    final_fault = args.final_fault if args.final_fault else "golden"
    dataset = args.dataset

    # prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), dataset + '/' + model_type)
    model_name = '%s.h5' % (final_fault)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=2,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # run training, with or without data augmentation.
    if not data_augment:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        steps_per_epoch =  math.ceil(len(x_train) / batch_size)
        # fit the model on the batches generated by datagen.flow().
        model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
                  verbose=2,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks)

def load_gtsrb():
    root_dir = "/home/GTSRB/"
    train_root_dir = root_dir + "Final_Training/Images/"
    test_root_dir = root_dir + "Final_Test/"

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


def get_trained_student(teacher, student, x_train, y_train, x_test, y_test):
    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=3,
    )

    # Distill teacher to student
    distiller.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate student on test dataset
    scores = distiller.evaluate(x_test, y_test, batch_size=batch_size)
    print('Student Test accuracy:', scores[0])
    print('Student Test loss:', scores[1])
    return distiller


def main():
    # model name, depth
    model_type = args.model_type
    fault_str = args.final_fault if args.final_fault else "golden"
    dataset_name = args.dataset

    (x_train, y_train), (x_test, y_test) = load_training_data(args.dataset)

    # input image dimensions.
    input_shape = x_train.shape[1:]

    teacher = get_model_by_name(model_type, input_shape)
    student = get_model_by_name(model_type, input_shape)

    get_trained_model(teacher, model_type, x_train, y_train, x_test, y_test, data_augmentation)

    # score teacher model
    scores = teacher.evaluate(x_test,
                              y_test,
                              batch_size=batch_size,
                              verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Self distillation begins
    student = get_trained_student(teacher, student, x_train, y_train, x_test, y_test)

    preds_test = student.predict(x_test)
    predictions = np.argmax(preds_test, axis=1)
    pred_list = predictions.tolist()

    with open("./injection/" + dataset_name + "/kd_self-" + model_type + "-" + fault_str + "-" + str(identifier), "w") as w_file:
        json.dump(pred_list, w_file)


if __name__ == "__main__":
    main()
