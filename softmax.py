# python3 softmax.py [day_ahead]


import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Lambda
from keras.models import Model
from keras.constraints import min_max_norm
import keras.optimizers as optimizers
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
from keras.constraints import min_max_norm
import pdb
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

window_size = 5
train_size=260

try:
    day_ahead = int(sys.argv[1])
except Exception as e:
    day_ahead = 1

if day_ahead == 1:
    end_skip_rows = 2800
elif day_ahead == 2:
    end_skip_rows = 2794
elif day_ahead == 5:
    end_skip_rows = 2776
else:
    end_skip_rows = 2674

dataset = pandas.read_csv('data_new/DowJones2.csv', skiprows=list(range(1, end_skip_rows)))['Close']
percent_inc = []
for i in range(1, len(dataset)):
    percent_inc.append((dataset[i]-dataset[i-1])/dataset[i])
percent_inc = numpy.array(percent_inc)

ranges = [-4, -3, -2, -1, 1, 2, 3, 4]
trend_inc = []
for i in range(len(percent_inc)):
    if percent_inc[i] < -0.005:
        trend_inc.append(-4)
    elif percent_inc[i] < -0.002:
        trend_inc.append(-3)
    elif percent_inc[i] < -0.001:
        trend_inc.append(-2)
    elif percent_inc[i] < 0:
        trend_inc.append(-1)
    elif percent_inc[i] < 0.001:
        trend_inc.append(1)
    elif percent_inc[i] < 0.002:
        trend_inc.append(2)
    elif percent_inc[i] < 0.005:
        trend_inc.append(3)
    else:
        trend_inc.append(4)

trend_inc = list(map(lambda x: x/4, trend_inc))
trend_inc = numpy.array(trend_inc).reshape(-1, 1)
percent_inc = percent_inc.reshape(-1, 1)

def create_dataset(dataset, look_back=1, day_ahead=1, train_size=50):
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(dataset) - look_back*day_ahead - day_ahead+1):
        x = dataset[i:i+look_back*day_ahead:day_ahead]
        if i < train_size:
            trainX.append(x)
            trainY.append(numpy.zeros(8))
            trainY[-1][ranges.index(int(dataset[i+look_back*day_ahead]*4))] = 1
        else:
            testX.append(x)
            testY.append(numpy.zeros(8))
            testY[-1][ranges.index(int(dataset[i+look_back*day_ahead]*4))] = 1
    return numpy.array(trainX), numpy.array(trainY), numpy.array(testX), numpy.array(testY)

def _evaluate(x, y):
    predicts = model.predict(x).tolist()
    global corrects
    corrects = []
    for i in range(len(predicts)):
        index_max_predict = predicts[i].index(max(predicts[i]))
        index_max_reality = y[i].tolist().index(max(y[i]))
        if int(index_max_predict/4) == int(index_max_reality/4):
            corrects.append(1)
        else:
            corrects.append(0)
    return numpy.mean(corrects)
def _evaluate_sigmoid(x, y):
    predicts = model.predict(x).reshape(-1).tolist()
    global corrects
    corrects = []
    for i in range(len(predicts)):
        if predicts[i]*y[i][0] >= 0:
            corrects.append(1)
        else:
            corrects.append(0)
    return numpy.mean(corrects)

results = []
# pdb.set_trace()
for runtime in range(1):
    train_x, train_y, test_x, test_y = create_dataset(trend_inc, window_size, day_ahead, train_size)

    model = Sequential()
    model.add(LSTM(32, input_shape=(window_size, 1), return_sequences=False))
    # model.add(LSTM(32, input_shape=(window_size, 1)))
    model.add(Dense(40))
    # model.add(Dropout(.2))
    model.add(Dense(8, activation='softmax'))
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['acc'])

    # hist = model.fit(train_x, train_y, epochs=1000, batch_size=40, verbose=0)
    # result = [runtime, 
    #           'Train',
    #             model.evaluate(train_x, train_y, verbose=0), 
    #             _evaluate(train_x, train_y), 
    #             'Test',
    #             model.evaluate(test_x, test_y, verbose=0),
    #             _evaluate(test_x, test_y)]
    # print(result)
    # results.append(result)

    for i in range(2000):
        hist = model.fit(train_x, train_y, epochs=1, batch_size=40, verbose=0)
        eva_train = model.evaluate(train_x, train_y, verbose=0)
        eva_test = model.evaluate(test_x, test_y, verbose=0)
        result = [i, 
            'Train',
            '%.4f' % eva_train[0], # Loss or accuracy predict correct groups on training set 
            '%.4f' % eva_train[1], # Loss or accuracy predict correct groups on training set 
            '%.4f' % _evaluate(train_x, train_y), # Trend on training set 
            'Test',
            '%.4f' % eva_test[0], # Loss or accuracy predict correct groups on testing set 
            '%.4f' % eva_test[1], # Loss or accuracy predict correct groups on testing set 
            '%.4f' % _evaluate(test_x, test_y)] # Trend on testing set
        print(result)



# conference
# 8 - 2018
# 3 - 2018
