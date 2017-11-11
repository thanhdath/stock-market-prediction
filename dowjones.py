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
# import sys
import matplotlib.dates as mdates
import datetime as dt
import time
import h5py
import keras.backend as K
from keras.models import load_model
import tensorflow as tf
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)


# data preparation and normalization
features = ['Close', 'Open', 'High', 'Low', 'Volume']
window_size = 22
train_size = 800

dataset = pandas.read_csv('data_new/DowJones.csv')[features]
indexes = dataset.index.values[::-1]
dataset.loc[indexes[0]] = [max(dataset[a]) * 2 for a in features]

def create_dataset(dataset, look_back=1, train_size=50):
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i+look_back), :]
        if i < train_size:
            trainX.append(x)
            trainY.append(numpy.array([dataset[i+look_back, features.index('Close')]]))
        else:
            testX.append(x)
            testY.append(numpy.array([dataset[i+look_back, features.index('Close')]]))
    return numpy.array(trainX), numpy.array(trainY), numpy.array(testX), numpy.array(testY)

test_size = len(dataset) - train_size
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
dataset = dataset[:-1]
train_x, train_y, test_x, test_y = create_dataset(dataset, window_size, train_size)


# create model, train, predict

predicts = []
running_time = []
models = []
batch_size = 40
size_pred = 20
model = Sequential()
model.add(LSTM(64, 
              input_shape=(window_size, len(features)), 
              return_sequences=True, 
              kernel_initializer='random_normal', 
              bias_initializer='random_normal',
              bias_constraint=min_max_norm(min_value=0, max_value=0)))
model.add(LSTM(64, 
              return_sequences=False, 
              kernel_initializer='random_normal',
              bias_initializer='random_normal', 
              input_shape=(window_size, len(features)),
              bias_constraint=min_max_norm(min_value=0, max_value=0)))
model.add(Dense(22, 
                activation='relu',
                bias_constraint=min_max_norm(min_value=0, max_value=0)))
model.add(Dense(1, bias_constraint=min_max_norm(min_value=0, max_value=0)))
model.compile(loss='mse', optimizer='adam')

for i in range(len(test_x)):
    start_time = time.time()
    verbose = 0
    if i%size_pred == 0:
        print(str(i) + '. Training...')
        hist = model.fit(train_x, train_y, epochs=50, batch_size=batch_size, verbose=0)
        print(hist.history['loss'][-1])
        
        predicts.append(model.predict(test_x[i:i+size_pred]).flatten())
        
        ps = model.predict(test_x[i:i+size_pred])
        todays = test_x[i:i+batch_size,-1, 0]
        trend_p = [int(numpy.sign(a - b)) for a,b in zip(ps.reshape(-1,1), todays)]

        trend_r = [int(numpy.sign(a - b)) for a,b in zip(test_y[i:i+size_pred].reshape(-1), todays)]
        trend = [1 if a == b else 0 for a, b in zip(trend_p, trend_r)]
        print(numpy.mean(trend))
        
        running_time.append(time.time() - start_time)
        print('time: ' + str(running_time[-1]))

    train_x = numpy.append(train_x, test_x[i].reshape(1, window_size, len(features)), axis=0)
    train_y = numpy.append(train_y, test_y[i].reshape(1, 1), axis=0)
    
predicts = numpy.array(predicts).reshape(-1, 1)
ps = []
for a in predicts:
    ps.append(a)
ps = numpy.concatenate(numpy.array(ps).flatten())
ps = ps.reshape(-1, 1)


# print total running time
print('Running time: ' + str(sum(running_time)))

# print trend

todays = test_x[:,-1, 0]
trend_p = [int(numpy.sign(a - b)) for a,b in zip(ps.reshape(-1), todays)]
# trend_p = [1]*test_y.shape[0]
trend_r = [int(numpy.sign(a - b)) for a,b in zip(test_y.reshape(-1), todays)]
trend = [1 if a == b else 0 for a, b in zip(trend_p, trend_r)]
print('Trend: ' + str(100*numpy.mean(trend)))

# print mse 

ps_trf, test_y_trf = ps, test_y
for i in range(len(features)):
    ps_trf = numpy.append(ps_trf, numpy.zeros((ps_trf.shape[0], 1)), axis=1)
    test_y_trf = numpy.append(test_y_trf, numpy.zeros((test_y_trf.shape[0], 1)), axis=1)
ps_trf = scaler.inverse_transform(ps_trf)[:,0]
test_y_trf = scaler.inverse_transform(test_y_trf)[:,0]
errors = abs(ps_trf - test_y_trf)
print('MSE: ' + str(numpy.mean(errors)))

# print mape

print('MAPE: ' + str(100*numpy.mean([error/ytrue for error, ytrue in zip(errors, test_y_trf)])))
