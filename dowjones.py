# python3 dowjones.py [day_ahead] [model_number]


import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Optimizer
from keras.constraints import min_max_norm
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.dates as mdates
import datetime as dt
import time
import h5py
import keras.backend as K
from keras.models import load_model
import tensorflow as tf
import pdb
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)


def model_1():
    model = Sequential()
    model.add(LSTM(32, 
                  input_shape=(window_size, len(features)), 
                  return_sequences=True, 
                  kernel_initializer='random_normal', 
                  bias_initializer='random_normal'))
    model.add(LSTM(32, 
                  return_sequences=False, 
                  kernel_initializer='random_normal',
                  bias_initializer='random_normal', 
                  input_shape=(window_size, len(features))))
    model.add(Dense(22, 
                    activation='relu'))
    model.add(Dense(1))
    return model 

def model_2():
    model = Sequential()
    model.add(LSTM(32, 
                  input_shape=(window_size, len(features)), 
                  return_sequences=False, 
                  kernel_initializer='random_normal', 
                  bias_initializer='random_normal'))
    model.add(Dense(22, 
                    activation='relu'))
    model.add(Dense(1))
    return model

def model_3():
    model = Sequential()
    model.add(LSTM(32, 
                  input_shape=(window_size, len(features)), 
                  return_sequences=False, 
                  kernel_initializer='random_normal', 
                  bias_initializer='random_normal'))
    model.add(Dense(1, activation='relu'))
    return model

# data preparation and normalization
features = ['Close', 'Open', 'High', 'Low', 'Volume']
window_size = 22

try:
    day_ahead = int(sys.argv[1])
except Exception as e:
    day_ahead = 1

try:
    if int(sys.argv[2]) == 2:
        model = model_2()
    elif int(sys.argv[2]) == 3:
        model = model_3()
    else:
        model = model_1()
except Exception as e:
    model = model_1()

if day_ahead == 1:
    train_size = 3043
elif day_ahead == 2:
    train_size = 3020
elif day_ahead == 5:
    train_size = 2951
else: 
    train_size = 2560

dataset = pandas.read_csv('data_new/DowJones2.csv')[features]
indexes = dataset.index.values[::-1]
# dataset.loc[indexes[0]] = [max(dataset[a]) * 2 for a in features]

def create_dataset(dataset, look_back=1, day_ahead=1, train_size=50):
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(dataset) - look_back*day_ahead - day_ahead+1):
        x = dataset[i:i+look_back*day_ahead:day_ahead, :]
        if i < train_size:
            trainX.append(x)
            trainY.append(numpy.array([dataset[i+look_back*day_ahead, features.index('Close')]]))
        else:
            testX.append(x)
            testY.append(numpy.array([dataset[i+look_back*day_ahead, features.index('Close')]]))
    return numpy.array(trainX), numpy.array(trainY), numpy.array(testX), numpy.array(testY)

# def create_dataset(dataset, look_back=1, day_ahead=1, train_size=50,):
#     trainX, trainY, testX, testY = [], [], [], []
#     for i in range(len(dataset) - look_back - day_ahead):
#         x = dataset[i:(i+look_back):day_ahead, :]
#         if i < train_size:
#             trainX.append(x)
#             trainY.append(numpy.array([dataset[i+look_back-1+day_ahead, features.index('Close')]]))
#         else:
#             testX.append(x)
#             testY.append(numpy.array([dataset[i+look_back-1+day_ahead, features.index('Close')]]))
#     return numpy.array(trainX), numpy.array(trainY), numpy.array(testX), numpy.array(testY)

def _evaluate(test_x, test_y):
    predicts = model.predict(test_x)
    todays = test_x[:,-1, 0]
    trend_p = [int(numpy.sign(a - b)) for a,b in zip(predicts.reshape(-1), todays)]
    trend_r = [int(numpy.sign(a - b)) for a,b in zip(test_y.reshape(-1), todays)]
    trend = [1 if a == b else 0 for a, b in zip(trend_p, trend_r)]
    return numpy.mean(trend)

print('Predict ', day_ahead, 'days')

test_size = len(dataset) - train_size
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
# dataset = dataset[:-1]
train_x, train_y, test_x, test_y = create_dataset(dataset, window_size, day_ahead, train_size)
# pdb.set_trace()
model.compile(loss='mse', optimizer='adam')
model.fit(train_x, train_y, epochs=50, batch_size=40, verbose=2)
print('Evaluate Train')
print(_evaluate(train_x, train_y))
print('Evaluate Test')
print(_evaluate(test_x, test_y))

# for i in range(2000):
#     hist = model.fit(train_x, train_y, epochs=1, batch_size=40, verbose=0)
#     eva_test = _evaluate(test_x, test_y)
#     eva_train = _evaluate(train_x, train_y)
#     print(i, 
#         '%.7f' % hist.history['loss'][-1],  # Mean squared error
#         '%.4f' % eva_train, # Trend on training set
#         '%.4f' % eva_test) # Trend on testing set

