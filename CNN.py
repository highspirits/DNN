# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 08:00:17 2017

@author: VSNalam
"""

#https://www.kaggle.com/toregil/welcome-to-deep-learning-cnn-99

import os
os.chdir("C:\Sreedhar\Analytics\Kaggle\MNIST\Code")
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers     import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks      import LearningRateScheduler

#load train and test datasets
train = pd.read_csv("C:/Sreedhar/Analytics/Kaggle/MNIST/Data/train.csv")
test  = pd.read_csv("C:/Sreedhar/Analytics/Kaggle/MNIST/Data/test.csv")

# split the data into a training set and a validation set, so that we can evaluate the performance 
#of our model.
x_train, x_val, y_train, y_val = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size = 0.1, random_state = 123)

#Each data point consists of 784 values. A fully connected net just treats all these values the same, 
#but a CNN treats it as a 28x28 square. 
#We now reshape all data in matrix format. Keras wants an extra dimension in the end, for channels.
# If this had been RGB images, there would have been 3 channels, but as MNIST is gray scale it only uses 
#one.

#convert pandas to numpy array and reshape for doing convolution
x_train = x_train.values
x_val   = x_val.values

x_train = x_train.reshape(-1, 28, 28, 1)
x_val   = x_val.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_val   = to_categorical(y_val)

#Pre-process the data
x_train = x_train.astype("float32")/255.
x_val   = x_val.astype("float32")/255.

#Building the Model
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
#Output size of feature map of above step = (26,26,1) (28-3 + 1)
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))

datagen = ImageDataGenerator(zoom_range = 0.1, height_shift_range = 0.1, width_shift_range = 0.1, rotation_range = 0.1)

model.compile(optimizer = Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#We train once with a smaller learning rate to ensure convergence. We then speed things up, 
#only to reduce the learning rate by 10% every epoch. Keras has a function for this:
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

#We will use a very small validation set during training to save time in the kernel.
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=20, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed
                           callbacks=[annealer])





