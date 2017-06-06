
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


# In[ ]:



