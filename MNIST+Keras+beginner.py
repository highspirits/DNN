
# coding: utf-8

# In[26]:

import os
#os.chdir("C:\Sreedhar\Software\Anaconda\Install\envs\dnn\Lib\site-packages")
import tensorflow as tf
import keras
#os.chdir("C:\Sreedhar\Software\Anaconda\Install\envs\dnn")
os.chdir("C:\Sreedhar\Analytics\Kaggle\MNIST\Code")


# In[3]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[1]:

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import adam, RMSprop
from sklearn.model_selection import train_test_split


# In[4]:

#load train and test datasets
train = pd.read_csv("C:/Sreedhar/Analytics/Kaggle/MNIST/Data/train.csv")
test  = pd.read_csv("C:/Sreedhar/Analytics/Kaggle/MNIST/Data/test.csv")


# In[8]:

#Analyze test and train datasets 
print(train.shape)
print(test.shape)


# In[45]:

#Seperate target column and pixel columns
x_train = (train.ix[:, 1:].values).astype(np.float32)
y_train = (train.ix[:, 0].values).astype(np.float32)
x_test = test.values.astype(np.float32)


# In[46]:

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(y_train.shape)
print(y_train)


# In[47]:

#pre-processing the digit images
# centre the data around zero mean and unit variance.
x_mean = x_train.mean().astype(np.float32)
x_std  = x_train.std().astype(np.float32)
x_train = (x_train - x_mean)/x_std

#One hot encoding of output labels
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
#shape of y_train = (42000, 10)


# In[56]:

#Design Neural Network
np.random.seed(43)
x_train = x_train.reshape(x_train.shape[0], 784)
print(x_train.shape)


# In[65]:

from keras.models      import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks   import EarlyStopping
from keras.layers      import BatchNormalization, Convolution2D, MaxPooling2D

model = Sequential()

model.add(Dense(10, activation = 'relu', input_shape = (x_train.shape[1], )))
model.add(Dense(10, activation= 'softmax'))
print("input shape", model.input_shape)
print("output shape", model.output_shape)


# In[67]:

#Compile the network
#model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = 0.01), metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#fit the model
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=5)

model.fit(x_train, y_train, validation_split=0.3, epochs = 50, callbacks = [early_stopping_monitor])
#model.fit(x_train, y_train, validation_split=0.3, epochs = 20)


# In[69]:

#reshape test data
x_test = x_test.reshape(x_test.shape[0], 784)


# In[71]:

#Make predictions
predictions = model.predict_classes(x_test)


# In[73]:

#output data
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("model1.csv", index=False, header=True)

