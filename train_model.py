#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import cv2
import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import keras

from os import listdir
from os.path import isfile, join

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


dataset_folder = "./dataset/"
files = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
print("[IMAGES LOADED]")

train_dog = "./dataset/train/dogs/"
train_cat = "./dataset/train/cats/"
val_dog = "./dataset/validation/dogs/"
val_cat = "./dataset/validation/cats/"

dirs = [train_dog, train_cat, val_dog, val_cat]

for dir in dirs:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

counter_dogs = 0
counter_cats = 0
train_size = 1000
test_size = 500
resize = 150
total = train_size + test_size

train_images = []
test_images = []
train_labels = []
test_labels = []

def amount_of_zeros(val):
    if(val < 10):
        return "00"
    elif(val > 10 and val < 100):
        return "0"
    else:
        return ""

for index, file in enumerate(files):
    my_path = dataset_folder + file
    if files[index][0:3] == "cat":
        counter_cats += 1

        photo = cv2.imread(my_path)
        photo = cv2.resize(photo, (resize, resize), interpolation = cv2.INTER_AREA)

        if counter_cats <= train_size:
            train_images.append(photo)
            train_labels.append(0)

            name = train_cat + "cat" + str(amount_of_zeros(counter_cats)) + str(counter_cats) + ".jpg"
            cv2.imwrite(name, photo)

        elif counter_cats > train_size and counter_cats <= total:
            test_images.append(photo)
            test_labels.append(0)

            name = val_cat + "cat" + str(amount_of_zeros(counter_cats-1000)) + str(counter_cats-1000) + ".jpg"
            cv2.imwrite(name, photo)

    elif files[index][0:3] == "dog":
        counter_dogs += 1

        photo_read = cv2.imread(my_path)
        photo = cv2.resize(photo_read, (resize, resize), interpolation = cv2.INTER_AREA)

        if counter_dogs > train_size and counter_dogs <= total:
            test_images.append(photo)
            test_labels.append(1)

            name = val_dog + "dog" + str(amount_of_zeros(counter_dogs-1000)) + str(counter_dogs-1000) + ".jpg"
            cv2.imwrite(name, photo)

        elif counter_dogs <= train_size:
            train_images.append(photo)
            train_labels.append(1)

            name = train_dog + "dog" + str(amount_of_zeros(counter_dogs)) + str(counter_dogs) + ".jpg"
            cv2.imwrite(name, photo)

    if counter_dogs == total and counter_cats == total:
        break

print("[TRAINING AND TESTING DATA EXTRACTED]")

X_train = np.asarray(train_images).astype('float32')
Y_train = np.asarray(train_labels)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
X_test = np.asarray(test_images).astype('float32')
Y_test = np.asarray(test_labels)
Y_test = Y_test.reshape(Y_test.shape[0], 1)

# Normalize
X_train /= 255
X_test /= 255

shape = (X_train[0].shape[0], X_train[1].shape[0], 3)
print(shape)

#DATA AUGMENTATION
batch_size = 16
shape = (150, 150, 3)
width = 150
height = 150

train_samples = 2000
validation_samples = 1000
batch_size = 16
###
epochs = 300

train_dir = './dataset/train'
validation_dir = './dataset/validation'

datagenerator_validation = ImageDataGenerator(rescale = 1./255)

datagenerator_train = ImageDataGenerator(
      rescale = 1./255,
      rotation_range = 30,
      width_shift_range = 0.3,
      height_shift_range = 0.3,
      horizontal_flip = True,
      fill_mode = 'nearest')

train_gen = datagenerator_train.flow_from_directory(
        train_dir,
        target_size = (width, height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True)

validation_gen = datagenerator_validation.flow_from_directory(
        validation_dir,
        target_size = (width, height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False)

print("[DATA AUGMENTED]")

clf = Sequential()
clf.add(Conv2D(32, (3, 3), input_shape=shape))
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))

clf.add(Conv2D(32, (3, 3)))
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))

clf.add(Conv2D(64, (3, 3)))
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))

clf.add(Flatten())
clf.add(Dense(64))
clf.add(Activation('relu'))
clf.add(Dropout(0.5))
clf.add(Dense(1))
clf.add(Activation('sigmoid'))

clf.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


history = clf.fit_generator(
    train_gen,
    steps_per_epoch = train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_gen,
    validation_steps = validation_samples // batch_size)

print("[MODEL TRAINED AND SAVED]")

history_dict = history.history

train_loss = history_dict['loss']
test_loss = history_dict['val_loss']
epochs = range(1, len(train_loss) + 1)

print("TRAIN LOSS: " + str(train_loss[-1]))
print("TEST LOSS: " + str(test_loss[-1]))

plot1 = plt.plot(epochs, test_loss, label='Test Loss')
plot2 = plt.plot(epochs, train_loss, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.setp(plot1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(plot2, linewidth=2.0, marker = '4', markersize=10.0)
plt.legend()
plt.grid(True)
plt.show()

history_dict = history.history

train_accuracies = history_dict['acc']
test_accuracies = history_dict['val_acc']
epochs = range(1, len(train_loss) + 1)

print("TRAIN ACCURACY: " + str(train_accuracies[-1]))
print("TEST ACCURACY: " + str(test_accuracies[-1]))

plot1 = plt.plot(epochs, test_accuracies, label='Test Accuracy')
plot2 = plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.setp(plot1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(plot2, linewidth=2.0, marker = '4', markersize=10.0)
plt.legend()
plt.grid(True)
plt.show()
