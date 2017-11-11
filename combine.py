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

# features = ['Open', 'High', 'Low', 'Close', 'Volume']
features = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
dataset = pandas.read_csv('data_new/VNIndex.csv')[features]

indexes = dataset.index.values[::-1]
# print(indexes)
dataset.index = indexes
dataset = dataset[::-1]
dataset.loc[indexes[0]] = [max(dataset[a]) * 2 for a in features]
# print(dataset)

# for f in features:
#     plt.figure(dpi=200)
#     plt.plot(dataset[f], linewidth=.5)
#     plt.show()

def create_dataset(dataset, look_back=1, train_size=50):
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i+look_back), :]
        if i <= train_size:
            trainX.append(x)
            trainY.append(numpy.array([dataset[i+look_back, features.index('CLOSE')]]))
        else:
            testX.append(x)
            testY.append(numpy.array([dataset[i+look_back, features.index('CLOSE')]]))
    return numpy.array(trainX), numpy.array(trainY), numpy.array(testX), numpy.array(testY)
window_size = 22
# train_size = int(len(dataset)*0.8)
# train_size = len(dataset)
train_size=767
test_size = len(dataset) - train_size

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
dataset = dataset[:-1]

train_x, train_y, test_x, test_y = create_dataset(dataset, window_size, train_size)

# one = tf.placeholder(tf.float32, shape=(1))
# zero = tf.placeholder(tf.float32, shape=(1))
# sess= tf.InteractiveSession()
# sess.run(one, feed_dict={one: [1], zero:[0]})

def custom_activation(x):
    return K.relu(x)
# def custom_loss(y_true, y_pred):
# #     prev_days = [0] + y_true[:-1]
#     global a 
#     global b 
#     a = y_true
#     b = y_pred
#     sess= tf.InteractiveSession()
#     tensor = tf.placeholder(tf.int32, shape=(None))
#     sess.run(tensor, feed_dict={tensor: list(range(K.int_shape(y_pred)[0] or -1))})
    
#     prev_days = [0] + y_true[0:-1]
    
# #     tensor = tf.constant(, dtype='int32')
#     print(tensor)
#     print(y_true)
#     print(y_pred)
#     print(prev_days)
#     trend_r = Lambda(lambda x:
#                      y_true[:,x]-
#                      prev_days[x], 
#                      output_shape=(None,))(tensor)
#     trend_p = Lambda(lambda x:y_pred[x]-prev_days[x])(tensor)
#     not_equal = Lambda(lambda x:one if trend_r[x] == trend_p[x] else zero)(tensor) 
#     return K.mean(K.square(y_pred - y_true), axis=-1) + K.sum(not_equal, axis=0)

def loss_fn(ytrue, ypred):
    nds = tf.concat([ytrue[1:], [[1]]], 0)
    one = tf.constant(list([-1.0] * 768))
    trendt = nds - ytrue
    trendp = nds - ypred
    trend = tf.div(tf.multiply(trendp, one), trendt)
    return K.mean(trend, axis=-1)

def trend(ytrue, ypred):
    prev_days = tf.concat([[[0]], ytrue[:-1]], 0)
    one = tf.constant(list([-1.0] * 768))
    trendt = K.sign(ytrue - prev_days)
    trendp = K.sign(ypred - prev_days)
    correct = K.equal(trendt, trendp)
    return K.mean(K.cast(correct, dtype='float32'))

def cal_trend(model):
    ps = model.predict(test_x)
    todays = test_x[:,-1, 0]
    trend_p = [int(numpy.sign(a - b)) for a,b in zip(ps.reshape(-1,1), todays)]
    # trend_p = [1]*test_y.shape[0]
    trend_r = [int(numpy.sign(a - b)) for a,b in zip(test_y.reshape(-1), todays)]
    trend = [1 if a == b else 0 for a, b in zip(trend_p, trend_r)]
    print('-----  Trend  ------')
    print(numpy.mean(trend))

def mse(ytrue, ypred):
    prev_days = tf.concat([[[0]], ytrue[:-1]], 0)
    twos = tf.constant(list([2.0]*768))
    comparet = tf.greater(ytrue, prev_days)
    comparep = tf.greater(ypred, prev_days)
    compare = tf.equal(comparet, comparep)
    print(compare)
    hs = Lambda(lambda x: tf.constant(0.0) if x == True else tf.constant(1.0))(compare)

    return K.mean(tf.multiply(hs, K.square(ypred-ytrue)), axis=-1)

model = Sequential()
model.add(LSTM(64, input_shape=(window_size, len(features)), return_sequences=True, 
               kernel_initializer='random_uniform',
              bias_initializer='zeros'))
model.add(LSTM(64, return_sequences=False, kernel_initializer='random_uniform',
               bias_initializer='zeros',
               input_shape=(window_size, len(features))))
# model.add(LSTM(64, return_sequences=False, kernel_initializer='random_uniform',
#               bias_initializer='zeros', input_shape=(1, 5)))
model.add(Dense(22, activation='relu'))
model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
model.compile(loss=mse, optimizer='adam', metrics=[trend])
# t = model.get_weights()
hist = model.fit(train_x, train_y, epochs=100,batch_size=768, verbose=1, shuffle=False)

print('----- Evaluate ------')
print(model.evaluate(test_x, test_y))

cal_trend(model)

fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300)

plt.subplot(1, 2, 1)
plt.plot(train_y)
plt.plot(model.predict(train_x))

plt.subplot(1, 2, 2)
plt.plot(test_y)
plt.plot(model.predict(test_x))

plt.show()
